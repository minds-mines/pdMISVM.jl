import MLJModelInterface
const MMI = MLJModelInterface

using LinearAlgebra: norm, I, pinv, tr, Diagonal
using Flux: onehotbatch, onecold
using KernelFunctions: GammaExponentialKernel, Kernel, kernelmatrix
using MLJ: levels, nrows
using CategoricalArrays: CategoricalArray
using TimerOutputs

mutable struct KernelpdMISVMClassifier <: MMI.Deterministic
    C::Float64
    kernel::String
    γ::Float64
    μ::Float64
    ρ::Float64
    maxiter::Int64
    tol::Float64
end

mutable struct KernelpdMISVMVariables
    # Original vars
    X::Array{Float64, 2}
    X_cut::Array{UnitRange{Int64}, 1}
    Y::Array{Float64, 2}
    WᵀW::Array{Float64, 2}
    WᵀΦX::Array{Float64, 2}
    WyᵀΦX::Array{Float64, 2}
    kernel::Kernel
    b::Array{Float64, 1}
    by::Array{Float64, 2}

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
    ΦhatᵀΦhat::Array{Array{Float64, 2}, 1}
    ΦhatᵀΦX::Array{Array{Float64, 2}, 1}
end

function init_vars(model::KernelpdMISVMClassifier, _X, _y)
    N = length(_X)
    K = length(levels(_y))

    X = hcat([MMI.matrix(x)' for x in _X]...)
    nis = [nrows(x) for x in _X]
    X_cut = [sum(nis[1:ni])-nis[ni]+1:sum(nis[1:ni]) for ni in 1:N]
    Y = onehotbatch(_y, levels(_y)) .* 2.0 .- 1.0
    d, NI = size(X)
    WᵀW = randn(K, K)
    WᵀΦX = randn(K, NI)
    WyᵀΦX = randn(K, NI)
    if model.kernel == "rbf"
        kernel = GammaExponentialKernel(γ=model.γ)
    end
    b = randn(K)
    by = randn(K, NI)

    # Introduced vars
    E = randn(K, N)
    Q = randn(K, N)
    R = randn(K, N)
    T = randn(K, NI)
    U = randn(K, NI)

    # Lagrangian Multipliers
    Λ = zeros(K, N)
    Σ = zeros(K, N)
    Ω = zeros(K, N)
    Θ = zeros(K, NI)
    Ξ = zeros(K, NI)

    # Auxilary vars
    μ = model.μ
    YI = hcat([repeat(Y[:,i], outer=(1, length(cut))) for (i, cut) in enumerate(X_cut)]...)
    ΦhatᵀΦhat = Array{Float64, 2}[]
    ΦhatᵀΦX = Array{Float64, 2}[]
    
    reset_timer!()
    for m in 1:K
        @timeit "calc prime" prime = YI[m,:] .> 0
        @timeit "calc XXprime" XXprime = hcat(X, X[:,prime])
        @timeit "calc ΦhatᵀΦhat kernel" push!(ΦhatᵀΦhat, kernelmatrix(kernel, XXprime))
        @timeit "calc ΦhatᵀΦX kernel" push!(ΦhatᵀΦX, kernelmatrix(kernel, XXprime, X))
    end
    #print_timer()

    v = KernelpdMISVMVariables(X, X_cut, Y, WᵀW, WᵀΦX, WyᵀΦX, kernel, b, by, E, Q, R, T, U, Λ, Σ, Ω, Θ, Ξ, μ, YI, ΦhatᵀΦhat, ΦhatᵀΦX)
    calc_by!(v)

    return v
end


function obj_loss(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    𝓛 = 0.0
    𝓛 += 0.5 * tr(v.WᵀW)
    𝓛 += model.C * sum(max.(1 .- (bag_max(v.WᵀΦX .+ v.b, v.X_cut) - bag_max(v.WyᵀΦX + v.by, v.X_cut)).*v.Y, 0))

    return 𝓛
end

function lagrangian_loss(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    𝓛 = 0.0
    𝓛 += 0.5 * tr(v.WᵀW)
    𝓛 += model.C * sum(max.(v.Y .* v.E, 0))
    𝓛 += 0.5 * v.μ * norm(v.E - v.Y + v.Q - v.R + v.Λ/v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.Q - bag_max(v.T, v.X_cut) + v.Σ/v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.T - (v.WᵀΦX .+ v.b) + v.Θ/v.μ)^2.0
    𝓛 += 0.5 * v.μ * norm(v.R - bag_max(v.U, v.X_cut) + v.Ω/v.μ)^2.0
    calc_by!(v)
    𝓛 += 0.5 * v.μ * norm(v.U - (v.WyᵀΦX + v.by) + v.Ξ/v.μ)^2.0

    return 𝓛
end

function bag_max(WX, X_cut)
    return hcat([maximum(WX[:, cut], dims=2) for cut in X_cut]...)
end

function calc_by!(v::KernelpdMISVMVariables)
    K, NI = size(v.YI)
    bm = repeat(v.b, outer=(1, NI))
    v.by = repeat(bm[v.YI .> 0]', outer=(K, 1))
end

function W_update_fast!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    K, NI = size(v.YI)

    s1 = (v.T .- v.b + v.Θ/v.μ)
    
    for m in 1:K
        @timeit "prime calc" prime = v.YI[m,:] .> 0
        @timeit "s2 calc" s2 = sum((v.U - v.by + v.Ξ/v.μ)[:,prime], dims=1)/K
        @timeit "S calc" S = [s1[m,:]' s2]
        @timeit "D calc" D = Diagonal(vcat(ones(NI), 1/K*ones(length(s2))))
        @timeit "A calc" A = v.ΦhatᵀΦhat[m] .+ D/v.μ
        @timeit "B create" B = copy(v.ΦhatᵀΦX[m])
        @timeit "LALU" LAPACK.gels!('N', A, B)
        #@timeit "LU" F = lu(v.ΦhatᵀΦhat[m] .+ D/v.μ)
        @timeit "WᵀΦX calc with left division" v.WᵀΦX[m,:] = S * B
        #@timeit "WᵀW calc" v.WᵀW[m,m] = (S * ΦhatᵀΦhatDinv * v.ΦhatᵀΦhat[m] * ΦhatᵀΦhatDinv * S')[1,1]
    end

    @timeit "WyTX calc" v.WyᵀΦX = repeat(v.WᵀΦX[v.YI .> 0]', outer=(K, 1))
end

function W_update!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    K, NI = size(v.YI)

    s1 = (v.T .- v.b + v.Θ/v.μ)
    
    for m in 1:K
        @timeit "prime calc" prime = v.YI[m,:] .> 0
        @timeit "s2 calc" s2 = sum((v.U - v.by + v.Ξ/v.μ)[:,prime], dims=1)/K
        @timeit "S calc" S = [s1[m,:]' s2]
        @timeit "D calc" D = Diagonal(vcat(ones(NI), 1/K*ones(length(s2))))
        #println("taking inverse of " * string(size(v.ΦhatᵀΦhat[m])) * " matrix")
        @timeit "ΦhatᵀΦhatDinv calc" ΦhatᵀΦhatDinv = inv(v.ΦhatᵀΦhat[m] .+ D/v.μ)
        @timeit "WᵀΦX calc" v.WᵀΦX[m,:] = S * ΦhatᵀΦhatDinv * v.ΦhatᵀΦX[m]
        @timeit "WᵀW calc" v.WᵀW[m,m] = (S * ΦhatᵀΦhatDinv * v.ΦhatᵀΦhat[m] * ΦhatᵀΦhatDinv' * S')[1,1]
    end

    @timeit "WyTX calc" v.WyᵀΦX = repeat(v.WᵀΦX[v.YI .> 0]', outer=(K, 1))
end


function b_update!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    K, NI = size(v.YI)

    numer1 = sum(v.T - v.WᵀΦX + v.Θ/v.μ, dims=2)
    numer2 = zeros(size(numer1))
    for m in 1:K
        prime = v.YI[m,:] .> 0
        numer2[m] = sum((v.U - v.WyᵀΦX + v.Ξ/v.μ)[:,prime])
    end
    numer = numer1 + numer2
    denom = float(NI .+ K * sum(v.YI .> 0.0, dims=2))

    v.b = vec(numer ./ denom)
    calc_by!(v)
end

function E_update!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    S = v.Y - v.Q + v.R - v.Λ/v.μ
    gt = (v.Y .* S) .> model.C/v.μ
    mid = 0 .<= (v.Y .* S) .<= model.C/v.μ

    v.E = S .* .!mid - gt .* v.Y .* (model.C/v.μ)
end

function Q_update!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    v.Q = 0.5 * (v.Y - v.E + v.R - v.Λ/v.μ + bag_max(v.T, v.X_cut) - v.Σ/v.μ)
end

function R_update!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    v.R = 0.5 * (v.E - v.Y + v.Q + v.Λ/v.μ + bag_max(v.U, v.X_cut) - v.Ω/v.μ)
end

function T_update!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    K = size(v.Y, 1)
    Φ = v.WᵀΦX .+ v.b - v.Θ/v.μ
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ϕᵢₘ = Φ[m, cut]
            v.T[m,cut] = ϕᵢₘ
            v.T[m,cut[1]+argmax(ϕᵢₘ)-1] = 0.5 * (maximum(ϕᵢₘ) + v.Q[m, i] + v.Σ[m, i]/v.μ)
        end
    end
end

function U_update!(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    K = size(v.Y, 1)
    Ψ = v.WyᵀΦX + v.by - v.Ξ/v.μ
    for (i, cut) in enumerate(v.X_cut)
        ni = length(cut)
        for m in 1:K
            ψᵢₘ = Ψ[m, cut]
            v.U[m,cut] = ψᵢₘ
            v.U[m,cut[1]+argmax(ψᵢₘ)-1] = 0.5 * (maximum(ψᵢₘ) + v.R[m, i] + v.Ω[m, i]/v.μ)
        end
    end
end

function calc_residuals(model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    E_res = v.E - (v.Y - v.Q + v.R)
    Q_res = v.Q - bag_max(v.T, v.X_cut)
    T_res = v.T - (v.WᵀΦX .+ v.b)
    R_res = v.R - bag_max(v.U, v.X_cut)
    U_res = v.U - (v.WyᵀΦX .+ v.by)

    return E_res, Q_res, T_res, R_res, U_res
end

function KernelpdMISVMClassifier(; C=1.0, kernel="rbf", γ=2.0, μ=1e-3, ρ=1.2, maxiter=1000, tol=1e-6)
    @assert all(i -> (i > 0), [C, μ, ρ, maxiter, tol])
    @assert 0.0 < γ <= 2.0
    @assert ρ > 1.0
    @assert kernel == "rbf"
    model = KernelpdMISVMClassifier(C, kernel, γ, μ, ρ, maxiter, tol)
end

function MMI.fit(model::KernelpdMISVMClassifier, verbosity::Integer, X, y)
    v = init_vars(model, X, y)

    if verbosity > 5
        E_res, Q_res, T_res, R_res, U_res = calc_residuals(model, v)
        res = sum([norm(r) for r in (E_res, Q_res, T_res, R_res, U_res)])

        ol = obj_loss(model, v)
        ll = lagrangian_loss(model, v)
        print("Loss: " * string(ol) * "     \t") 
        print("Lagrangian: " * string(ll) * "     \t") 
        println("Residual: " * string(res))
    end

    for i in 1:model.maxiter
        reset_timer!()
        @timeit "W update" W_update!(model, v)
        @timeit "b update" b_update!(model, v)
        @timeit "E update" E_update!(model, v)
        @timeit "Q update" Q_update!(model, v)
        @timeit "R update" R_update!(model, v)
        @timeit "T update" T_update!(model, v)
        @timeit "U update" U_update!(model, v)

        @timeit "calc residuals" E_res, Q_res, T_res, R_res, U_res = calc_residuals(model, v)

        @timeit "Λ update" v.Λ += v.μ * E_res
        @timeit "Σ update" v.Σ += v.μ * Q_res
        @timeit "Θ update" v.Θ += v.μ * T_res 
        @timeit "Ω update" v.Ω += v.μ * R_res
        @timeit "Ξ update" v.Ξ += v.μ * U_res

        res = sum([norm(r) for r in (E_res, Q_res, T_res, R_res, U_res)])

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

        if verbosity > 5
            print_timer()
        end
    end

    fitresult = (v.X, v.kernel, v.YI, v.T, v.Θ, v.U, v.Ξ, v.μ, v.b, v.by, y)
    cache = missing
    report = missing

    return fitresult, cache, report
end

function MMI.predict(model::KernelpdMISVMClassifier, fitresult, _Xnew)
    X, kernel, YI, T, Θ, U, Ξ, μ, b, by, y = fitresult

    K, NI = size(YI)
    N = length(_Xnew)
    Xnew = hcat([MMI.matrix(x)' for x in _Xnew]...)
    nis = [nrows(x) for x in _Xnew]
    X_cut = [sum(nis[1:ni])-nis[ni]+1:sum(nis[1:ni]) for ni in 1:N]

    WᵀΦXnew = zeros(K, sum(nis))
    s1 = (T .- b + Θ/μ)
    for m in 1:K
        prime = YI[m,:] .> 0
        s2 = sum((U - by + Ξ/μ)[:,prime], dims=1)/K
        S = [s1[m,:]' s2]
        D = Diagonal(vcat(ones(NI), 1/K*ones(length(s2))))
        XXprime = hcat(X, X[:,prime])

        ΦhatᵀΦhat = kernelmatrix(kernel, XXprime)
        ΦhatᵀΦhatDinv = inv(ΦhatᵀΦhat .+ D/μ)
        ΦhatᵀΦXnew = kernelmatrix(kernel, XXprime, Xnew)

        WᵀΦXnew[m,:] = S * ΦhatᵀΦhatDinv * ΦhatᵀΦXnew
    end

    raw_pred = bag_max(WᵀΦXnew .+ b, X_cut)

    pred = CategoricalArray(onecold(raw_pred, levels(y)))

    return pred
end
