import MLJBase
using LinearAlgebra: norm, I, pinv, eigen, Diagonal, mul!
using Flux: onehotbatch, onecold
using MLJ: levels, nrows
using CategoricalArrays: CategoricalArray
using TimerOutputs

mutable struct pdMISVMClassifier <: MLJBase.Deterministic
    C::Float64
    μ::Float64
    ρ::Float64
    maxiter::Int64
    tol::Float64
    exact::Bool
end

mutable struct misvm_vars
    # Original vars
    X::Array{Float64, 2}
    X_cut::Array{UnitRange{Int64}, 1}
    Y::Array{Float64, 2}
    W::Array{Float64, 2}
    b::Array{Float64, 1}

    # Introduced vars
    E::Array{Float64, 2}
    Q::Array{Float64, 2}
    R::Array{Float64, 2}
    T::Array{Float64, 2}
    U::Array{Float64, 2}

    # Lagrangian Multipliers
    Λ::Array{Float64, 2}
    Σ::Array{Float64, 2}
    Ω::Array{Float64, 2}
    Θ::Array{Float64, 2}
    Ξ::Array{Float64, 2}

    # Auxilary vars
    μ::Float64
    YI::Array{Float64, 2}
    WmX::Array{Float64, 2}
    WyX::Array{Float64, 2}
    by::Array{Float64, 2}

    # Note: the residual arrays can also be used as temporary storage
    E_res::Array{Float64, 2}
    Q_res::Array{Float64, 2}
    R_res::Array{Float64, 2}
    T_res::Array{Float64, 2}
    U_res::Array{Float64, 2}

    # To speed up W update (https://math.stackexchange.com/questions/670649/efficient-diagonal-update-of-matrix-inverse)
    Ps::Array{Array{Float64, 2}}
    Ds::Array{Diagonal}
    P⁻¹s::Array{Array{Float64, 2}}

    # Temporary arrays
    tmp1d::Array{Float64, 2} # A temporary 1 × d Array
    tmpdd::Array{Float64, 2} # A temporary d × d Array
    tmpdd2::Array{Float64, 2} # A second temporary d × d Array
    tmpKd::Array{Float64, 2} # A temporary K × d Array
    tmpKd2::Array{Float64, 2} # A second temporary K × d Array
    tmpKNI::Array{Float64, 2} # A temporary K × NI Array
    tmpKprime::Array{Array{Float64, 2}} # K temporary K × NIprime Arrays

    # vars for inexact W update
    s::Array{Float64, 1}
    ∇W::Array{Float64, 2}
    prime_sum::Array{Int64, 1}
    prime_cut::Array{UnitRange{Int64}, 1}
    Xᵀprime::Array{Array{Float64, 2}}
end

function misvm_vars(; X, X_cut, Y, W, b, E, Q, R, T, U, Λ, Σ, Ω, Θ, Ξ, μ, YI, WmX, WyX, by, E_res, Q_res, R_res, T_res, U_res, Ps, Ds, P⁻¹s, tmp1d, tmpdd, tmpdd2, tmpKd, tmpKd2, tmpKNI, tmpKprime, s, ∇W, prime_sum, prime_cut, Xᵀprime)
    v = misvm_vars(X, X_cut, Y, W, b, E, Q, R, T, U, Λ, Σ, Ω, Θ, Ξ, μ, YI, WmX, WyX, by, E_res, Q_res, R_res, T_res, U_res, Ps, Ds, P⁻¹s, tmp1d, tmpdd, tmpdd2, tmpKd, tmpKd2, tmpKNI, tmpKprime, s, ∇W, prime_sum, prime_cut, Xᵀprime)
end

function init_vars(model::pdMISVMClassifier, _X, _y)
    N = length(_X)
    K = length(levels(_y))

    # Sort bags by class to ensure regular memory access pattern during prime indexing (see prime_cut)
    perm = sortperm(_y)
    sort_y = _y[perm]
    sort_X = _X[perm]

    # Build the X matrix by concatenating all bags horizontally
    X = hcat([MLJBase.matrix(x)' for x in sort_X]...)
    d, NI = size(X)

    # Calculate the horizontal "cuts" to extract the nᵢ'th bag
    nis = [size(x, 1) for x in sort_X]
    X_cut = [sum(nis[1:ni])-nis[ni]+1:sum(nis[1:ni]) for ni in 1:N]

    # Build the y-hot matrix, hyperplanes and intercepts
    Y = onehotbatch(sort_y, levels(sort_y)) .* 2.0 .- 1.0
    W = randn(d, K)
    b = randn(K)

    # Build introduced variables
    E = randn(K, N)
    Q = randn(K, N)
    R = randn(K, N)
    T = randn(K, NI)
    U = randn(K, NI) 

    # Build Lagrangian multipliers
    Λ = zeros(K, N)
    Σ = zeros(K, N)
    Ω = zeros(K, N)
    Θ = zeros(K, NI)
    Ξ = zeros(K, NI)

    # Auxilary vars
    μ = model.μ
    YI = hcat([repeat(Y[:,i], outer=(1, length(cut))) for (i, cut) in zip(1:N, X_cut)]...)
    WmX = randn(K, NI)
    WyX = randn(K, NI)
    by = randn(K, NI)
    E_res = zeros(size(E))
    Q_res = zeros(size(Q))
    R_res = zeros(size(R))
    T_res = zeros(size(T))
    U_res = zeros(size(U))

    if model.exact
        rhs1 = sum([X[:,cut]*X[:,cut]' for cut in X_cut])
        rhs2 = [zeros(d, d) for i in 1:K]
        for m in 1:K
            step1 = [X[:,cut][:,YI[:,cut][m,:] .> 0] for cut in X_cut]
            rhs2[m] = sum([x * x' for x in step1])
        end
        As = [rhs1 + K*r2 for r2 in rhs2]
        eigAs = [eigen(A) for A in As]

        Ps = [real(eigA.vectors) for eigA in eigAs]
        Ds = [Diagonal(real(eigA.values)) for eigA in eigAs]
        P⁻¹s = [real(inv(eigA.vectors)) for eigA in eigAs]
    else
        Ps = [randn(1, 1)]
        Ds = [Diagonal(randn(1))]
        P⁻¹s = [randn(1, 1)]
    end

    # Inexact vars
    s = randn(K)
    ∇W = randn(size(W))
    prime_sum = vec(sum(YI .> 0, dims=2))
    prime_cut = [sum(prime_sum[1:m])-sum(prime_sum[m])+1:sum(prime_sum[1:m]) for m in 1:K] # Calculate the prime cut indicies
    Xᵀprime = [X[:,cut]' for cut in prime_cut]

    # Temporary arrays
    tmp1d = zeros(1, d)
    tmpdd = zeros(d, d)
    tmpdd2 = zeros(d, d)
    tmpKd = zeros(K, d)
    tmpKd2 = zeros(K, d)
    tmpKNI = zeros(K, NI)
    tmpKprime = [zeros(size(p, 1), 1) for p in Xᵀprime]

    v = misvm_vars(X=X, X_cut=X_cut, Y=Y, W=W, b=b, E=E, Q=Q, R=R, T=T, U=U, Λ=Λ, Σ=Σ, Ω=Ω, Θ=Θ, Ξ=Ξ, μ=μ, YI=YI, WmX=WmX, WyX=WyX, by=by, E_res=E_res, Q_res=Q_res, R_res=R_res, T_res=T_res, U_res=U_res, Ps=Ps, Ds=Ds, P⁻¹s=P⁻¹s, tmp1d=tmp1d, tmpdd=tmpdd, tmpdd2=tmpdd2, tmpKd=tmpKd, tmpKd2=tmpKd2, tmpKNI=tmpKNI, tmpKprime=tmpKprime, s=s, ∇W=∇W, prime_sum=prime_sum, prime_cut=prime_cut, Xᵀprime=Xᵀprime)

    calc_WmX_WyX!(v)
    calc_by!(v)

    return v
end

function obj_loss(model::pdMISVMClassifier, v::misvm_vars)
    𝓛 = 0.0
    𝓛 += 0.5 * norm(v.W, 2) ^ 2.0
    𝓛 += model.C * sum(max.(1 .- (bag_max!(v.Q_res, v.WmX .+ v.b, v.X_cut) .- bag_max!(v.R_res, v.WyX .+ v.by, v.X_cut)).*v.Y, 0))

    return 𝓛
end

function lagrangian_loss(model::pdMISVMClassifier, v::misvm_vars)
    𝓛 = 0.0
    𝓛 += 0.5 * norm(v.W, 2)^2.0
    𝓛 += model.C * sum(max.(v.Y .* v.E, 0))
    𝓛 += 0.5 * v.μ * norm(v.E .- v.Y .+ v.Q .- v.R .+ v.Λ./v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.Q .- bag_max!(v.Q_res, v.T, v.X_cut) .+ v.Σ./v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.T .- (v.WmX .+ v.b) .+ v.Θ./v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.R .- bag_max!(v.R_res, v.U, v.X_cut) .+ v.Ω./v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.U .- (v.WyX .+ v.by) .+ v.Ξ./v.μ)^2.0

    return 𝓛
end

function inexact_loss(model::pdMISVMClassifier, v::misvm_vars)
    d = size(v.W, 1)
    s = repeat(v.s', outer=(d, 1))

    newW = v.W - s.*v.∇W
    newWX = newW' * v.X
    K = size(v.Y, 1)
    newWyX = repeat(newWX[v.YI .> 0]', outer=(K, 1))

    𝓛 = 0.0
    𝓛 += 0.5 * norm(newW, 2)^2.0
    𝓛 += 0.5 * v.μ * norm(v.T .- (newWX .+ v.b) .+ v.Θ/v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.U .- (newWyX + v.by) .+ v.Ξ/v.μ)^2.0
end

function bag_max!(R, WX, X_cut)
    for (i, cut) in enumerate(X_cut)
        c = @view WX[:, cut] 
        r = @view R[:,i]
        maximum!(r, c)
    end
    return R
end

function calc_WmX_WyX!(v::misvm_vars)
    mul!(v.WmX, v.W', v.X)
    K = size(v.Y, 1)
    v.tmpKNI .= v.WmX .* (v.YI .> 0)
    for m in 1:K
        tmp = @view v.WyX[m:m,:]
        sum!(tmp, v.tmpKNI)
    end
end

function calc_by!(v::misvm_vars)
    for m in 1:size(v.Y, 1)
        bym = @view v.by[:, v.prime_cut[m]]
        bym .= v.b[m]
    end
end

function W_update!(model::pdMISVMClassifier, v::misvm_vars)
    K, NI = size(v.T)
    d = size(v.W, 1)

    @. v.T_res = v.T - v.b + v.Θ/v.μ; T̂ = @view v.T_res[:,:]
    @. v.U_res = v.U - v.by + v.Ξ/v.μ; Û = @view v.U_res[:,:]
    mul!(v.tmpKd, T̂, v.X')

    for m in 1:K
        Ûprime = @view Û[:,v.prime_cut[m]]; t̂Xᵀ = @view v.tmpKd[m,:]; w = @view v.W[:,m:m]
        mul!(v.tmpKd2, Ûprime, v.Xᵀprime[m])
        v.tmpdd2 .= inv(v.Ds[m] + I./v.μ)
        mul!(v.tmpdd, v.Ps[m], v.tmpdd2)
        sum!(v.tmp1d, v.tmpKd2)
        @. v.tmp1d += t̂Xᵀ'
        mul!(v.tmpdd2, v.tmpdd, v.P⁻¹s[m])
        mul!(w', v.tmp1d, v.tmpdd2)
    end

    calc_WmX_WyX!(v)
end

function inexact_W_update!(model::pdMISVMClassifier, v::misvm_vars)
    s_update!(model::pdMISVMClassifier, v::misvm_vars)

    @. v.W -= v.∇W .* v.s'

    calc_WmX_WyX!(v)
end

function s_update!(model::pdMISVMClassifier, v::misvm_vars)
    K, NI = size(v.T)

    v.T_res .= v.T .- v.WmX .- v.b .+ v.Θ./v.μ; T̂ = @view v.T_res[:,:]
    v.U_res .= v.U .- v.WyX .- v.by .+ v.Ξ./v.μ; Û = @view v.U_res[:,:]
    mul!(v.tmpKd, T̂,  v.X')
    v.tmpKd .= v.W' .- v.μ .* v.tmpKd

    for m in 1:K
        Ûprime = @view Û[:,v.prime_cut[m]]; tmp1d = @view v.tmpKd[m,:]; w = @view v.W[:,m]
        mul!(v.tmpKd2, Ûprime, v.Xᵀprime[m]); ÛXᵀprime = @view v.tmpKd2[:,:]
        ∇w = @view v.∇W[:,m]; sum!(∇w, ÛXᵀprime'); ∇w .= tmp1d .- v.μ .* ∇w
        Xᵀ∇w = @view v.tmpKNI[m,:]; mul!(Xᵀ∇w, v.X', ∇w); t̂ = @view T̂[m,:]
        numer = w' * ∇w .- v.μ * (t̂' * Xᵀ∇w) .- v.μ * sum(ÛXᵀprime * ∇w, dims=1)
        Xᵀprime∇w = @view v.tmpKprime[m][:,:]
        mul!(Xᵀprime∇w, v.Xᵀprime[m], ∇w)
        denom = norm(∇w, 2)^2.0 .+ v.μ.*norm(Xᵀ∇w)^2.0 .+ v.μ.*K.*norm(Xᵀprime∇w)^2.0
        v.s[m] = numer[1] / denom
    end
end

function b_update!(model::pdMISVMClassifier, v::misvm_vars)
    K, NI = size(v.YI)

    @. v.T_res = v.T - v.WmX + v.Θ/v.μ; T̂ = @view v.T_res[:,:]
    @. v.U_res = v.U - v.WyX + v.Ξ/v.μ; Û = @view v.U_res[:,:]
    sum!(v.b, T̂)
    for m in 1:K
        Ûprime = @view Û[:,v.prime_cut[m]]
        v.b[m] += sum(Ûprime)
    end
    v.b .= v.b ./ (NI .+ K .* v.prime_sum)

    calc_by!(v)
end

function E_update!(model::pdMISVMClassifier, v::misvm_vars)
    S = @view v.E_res[:,:]; S .= v.Y .- v.Q .+ v.R .- v.Λ./v.μ
    YS = @view v.Q_res[:,:]; YS .= v.Y .* S
    gt = YS .> model.C/v.μ
    mid = 0 .<= YS .<= model.C/v.μ

    v.E .= S .* .!mid .- gt .* v.Y .* (model.C/v.μ)
end

function Q_update!(model::pdMISVMClassifier, v::misvm_vars)
    bag_max!(v.Q_res, v.T, v.X_cut); bag_max_T = @view v.Q_res[:,:]
    v.Q .= 0.5 .* (v.Y .- v.E .+ v.R .- v.Λ./v.μ .+ bag_max_T .- v.Σ./v.μ)
end

function R_update!(model::pdMISVMClassifier, v::misvm_vars)
    bag_max!(v.R_res, v.U, v.X_cut); bag_max_U = @view v.R_res[:,:]
    v.R .= 0.5 .* (v.E .- v.Y .+ v.Q .+ v.Λ./v.μ .+ bag_max_U .- v.Ω./v.μ)
end

function T_update!(model::pdMISVMClassifier, v::misvm_vars)
    K = size(v.Y, 1)
    v.T_res .= v.WmX .+ v.b .- v.Θ./v.μ # Store data in T_res to save allocations
    Φ = @view v.T_res[:,:]
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ϕᵢₘ = @view Φ[m, cut]
            v.T[m,cut] = ϕᵢₘ
            v.T[m,cut[1]+argmax(ϕᵢₘ)-1] = 0.5 * (maximum(ϕᵢₘ) + v.Q[m, i] + v.Σ[m, i]/v.μ)
        end
    end
end

function U_update!(model::pdMISVMClassifier, v::misvm_vars)
    K = size(v.Y, 1)
    v.U_res .= v.WyX .+ v.by .- v.Ξ./v.μ # Store data in U_res to save allocations
    Ψ = @view v.U_res[:,:]
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ψᵢₘ = @view Ψ[m, cut]
            v.U[m,cut] = ψᵢₘ
            v.U[m,cut[1]+argmax(ψᵢₘ)-1] = 0.5 * (maximum(ψᵢₘ) + v.R[m, i] + v.Ω[m, i]/v.μ)
        end
    end
end

function calc_residuals!(model::pdMISVMClassifier, v::misvm_vars)
    v.E_res .= v.E .- (v.Y .- v.Q .+ v.R)
    v.Q_res .= v.Q .- bag_max!(v.Q_res, v.T, v.X_cut)
    v.T_res .= v.T .- (v.WmX .+ v.b)
    v.R_res .= v.R .- bag_max!(v.R_res, v.U, v.X_cut)
    v.U_res .= v.U .- (v.WyX .+ v.by)
end

function pdMISVMClassifier(; C=1.0, μ=1e-3, ρ=1.2, maxiter=1000, tol=1e-6, exact=true)
    @assert all(i -> (i > 0), [C, μ, ρ, maxiter, tol])
    @assert ρ > 1.0
    model = pdMISVMClassifier(C, μ, ρ, maxiter, tol, exact)
end

function MLJBase.fit(model::pdMISVMClassifier, verbosity::Integer, X, y)
    v = init_vars(model, X, y)

    if verbosity > 5
        calc_residuals!(model, v)
        res = sum([norm(r) for r in (v.E_res, v.Q_res, v.T_res, v.R_res, v.U_res)])

        ol = obj_loss(model, v)
        ll = lagrangian_loss(model, v)
        print("Loss: " * string(ol) * "     \t") 
        print("Lagrangian: " * string(ll) * "     \t") 
        println("Residual: " * string(res))
    end

    for i in 1:model.maxiter
        if model.exact
            W_update!(model, v)
        else
            inexact_W_update!(model, v)
        end
        b_update!(model, v)
        E_update!(model, v)
        Q_update!(model, v)
        R_update!(model, v)
        T_update!(model, v)
        U_update!(model, v)

        calc_residuals!(model, v)

        @. v.Λ += v.μ * v.E_res
        @. v.Σ += v.μ * v.Q_res
        @. v.Θ += v.μ * v.T_res 
        @. v.Ω += v.μ * v.R_res
        @. v.Ξ += v.μ * v.U_res
         
        res = sum([norm(r) for r in (v.E_res, v.Q_res, v.T_res, v.R_res, v.U_res)])

        if verbosity > 5
            ol = obj_loss(model, v)
            ll = lagrangian_loss(model, v)
            print("Loss: " * string(ol) * "     \t")
            print("Lagrangian: " * string(ll) * "     \t")
            println("Residual: " * string(res))
        end

        if res < model.tol
            break
        end

        v.μ = model.ρ * v.μ
    end

    fitresult = v.W, v.b, levels(y)
    cache = missing
    report = missing

    return fitresult, cache, report
end

function MLJBase.predict(model::pdMISVMClassifier, fitresult, Xnew)
    N = length(Xnew)
    W, b, levels_of_y = fitresult

    X = hcat([MLJBase.matrix(x)' for x in Xnew]...)
    nis = [size(x, 1) for x in Xnew]
    X_cut = [sum(nis[1:ni])-nis[ni]+1:sum(nis[1:ni]) for ni in 1:N]
    raw_pred = zeros(size(W, 2), N)
    bag_max!(raw_pred, W' * X .+ b, X_cut)

    pred = CategoricalArray(onecold(raw_pred, levels_of_y))

    return pred
end
