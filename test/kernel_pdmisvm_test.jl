using DrWatson
using Test
using MLJ

include(srcdir("KernelpdMISVMClassifier.jl"))

function init_dummy_data()
    bag1_3instance = [1.0  1.0  100.0;
                      1.0  2.0  0.0;
                      0.0  0.0  0.0]
    bag2_2instance = [1.0  2.0  100.0;
                      2.0  2.0 -2.0]
    bag3_1instance = [3.0  3.0 -100.0]
    bag4_1instance = [4.0  4.0 -100.0]

    X = [bag1_3instance, bag2_2instance, bag3_1instance, bag4_1instance]
    y = vec([0 0 1 1])

    return X, y
end

function init_dummy_kernel_misvm_and_var()
    model = KernelpdMISVMClassifier()
    X, y = init_dummy_data()

    v = init_vars(model, X, y)

    v.Λ .= 1.0; v.Σ .= 2.0; v.Ω .= 3.0; v.Θ .= 4.0; v.Ξ .= 5.0; v.μ = 2.0
	
    return model, v
end

@testset "check definitions" begin
    K = 2; D = 3; NI = 7; NIprime = 3; μ = 2

    Φ = randn(D, NI)
    Φprime = randn(D, NIprime)
    T = randn(1, NI)
    U = randn(1, NIprime)

    A = inv(I/μ + Φ*Φ' + K*Φprime*Φprime') * (Φ*T' + Φprime*U')
    @test size(A) == (D, 1)

    Φhat = hcat(Φ, Φprime)
    Dmat = Diagonal([ones(NI); K*ones(NIprime)])

    @test Φ*Φ'+K*Φprime*Φprime' ≈ Φhat*Dmat*Φhat'

    S = [T U/K]

    @test Φ*T'+Φprime*U' ≈ Φhat*Dmat*S'

    B = inv(I/μ + Φhat*Dmat*Φhat') * Φhat*Dmat*S'

    @test A ≈ B

    C = Φhat * inv(Φhat'*Φhat + inv(Dmat)/μ) * S'

    @test A ≈ C
end

@testset "check init" begin
    model, v = init_dummy_kernel_misvm_and_var()

    @test size(v.X) == (3, 7)
    @test v.X_cut == [1:3, 4:5, 6:6, 7:7]
    @test v.Y == [ 1.0  1.0 -1.0 -1.0;
                  -1.0 -1.0  1.0  1.0]
    @test size(v.WᵀW) == (2, 2)
    @test size(v.WᵀΦX) == (2, 7)
    @test size(v.WyᵀΦX) == (2, 7)
    @test size(v.b) == (2,)
    @test size(v.by) == (2, 7)
    
    @test size(v.E) == (2, 4)
    @test size(v.Q) == (2, 4)
    @test size(v.R) == (2, 4)
    @test size(v.T) == (2, 7)
    @test size(v.U) == (2, 7)

    @test size(v.Λ) == (2, 4)
    @test size(v.Σ) == (2, 4)
    @test size(v.Ω) == (2, 4)
    @test size(v.Θ) == (2, 7)
    @test size(v.Ξ) == (2, 7)
end

@testset "check losses" begin
    @testset "check obj loss" begin
        model, v = init_dummy_kernel_misvm_and_var()
        @test obj_loss(model, v) > 0.0
    end

    @testset "check lagrangian loss" begin
        model, v = init_dummy_kernel_misvm_and_var()
        @test lagrangian_loss(model, v) > 0.0
    end
end

function get_plus_minus_evals(var_to_check, model::KernelpdMISVMClassifier, v::KernelpdMISVMVariables)
    δ = .0001

    lower_bound = lagrangian_loss(model, v)
    var_to_check .+= δ
    plus_cost = lagrangian_loss(model, v)
    var_to_check .-= 2.0*δ
    minus_cost = lagrangian_loss(model, v)

    return lower_bound, minus_cost, plus_cost
end

@testset "check updates" begin
    @testset "check W update" begin
        model, v = init_dummy_kernel_misvm_and_var()
        l1 = lagrangian_loss(model, v)
        W_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        δ = .0001
        v.T .+= δ
        W_update!(model, v)
        v.T .-= δ
        plus_cost = lagrangian_loss(model, v)
        v.T .-= δ
        W_update!(model, v)
        v.T .+= δ
        minus_cost = lagrangian_loss(model, v)
        @test l2 < minus_cost
        @test l2 < plus_cost
    end

    @testset "check b update" begin
        model, v = init_dummy_kernel_misvm_and_var()
        l1 = lagrangian_loss(model, v)
        b_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.b, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check E update" begin
        model, v = init_dummy_kernel_misvm_and_var()
        l1 = lagrangian_loss(model, v)
        E_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.E, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check Q update" begin
        model, v = init_dummy_kernel_misvm_and_var()
        l1 = lagrangian_loss(model, v)
        Q_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.Q, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check R update" begin
        model, v = init_dummy_kernel_misvm_and_var()
        l1 = lagrangian_loss(model, v)
        R_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.R, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check T update" begin
        model, v = init_dummy_kernel_misvm_and_var()
        l1 = lagrangian_loss(model, v)
        T_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.T, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check U update" begin
        model, v = init_dummy_kernel_misvm_and_var()
        l1 = lagrangian_loss(model, v)
        U_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.U, model, v)
        @test lower < minus
        @test lower < plus
    end
end

@testset "check ml functions" begin
    @testset "check fit and predict" begin
        model = KernelpdMISVMClassifier()
        X, y = init_dummy_data()
        kernel_misvm = machine(model, X, y)

        fit!(kernel_misvm, verbosity=1)
        pred = predict(kernel_misvm, X)

        @test pred == y
    end
end
