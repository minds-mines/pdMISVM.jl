using DrWatson
using Test
using MLJ

include(srcdir("pdMISVMClassifier.jl"))

function init_dummy_data()
    bag1_3instance = [1.0  1.0  3.0;
                      1.0  2.0  0.0;
                      0.0  0.0  0.0]
    bag2_2instance = [1.0  2.0  4.0;
                      2.0  2.0 -2.0]
    bag3_1instance = [3.0  3.0 -5.0]
    bag4_1instance = [4.0  4.0 -2.0]

    X = [bag1_3instance, bag2_2instance, bag3_1instance, bag4_1instance]
    y = vec([0 0 1 1])

    return X, y
end

function init_dummy_model_and_var()
    model = pdMISVMClassifier(C=10.0)
    X, y = init_dummy_data()

    v = init_vars(model, X, y)

    # Set Lagrangian multiplier values and μ to something other than 0
    v.Λ .= 1.0; v.Σ .= 2.0; v.Ω .= 3.0; v.Θ .= 4.0; v.Ξ .= 5.0; v.μ = 2.0

    calc_WmX_WyX!(v)
    calc_by!(v)
	
    return model, v
end

@testset "check init" begin
    model, v = init_dummy_model_and_var()

    @test size(v.X) == (3, 7)
    @test v.X_cut == [1:3, 4:5, 6:6, 7:7]
    @test v.Y == [ 1.0  1.0 -1.0 -1.0;
                  -1.0 -1.0  1.0  1.0]
    @test size(v.W) == (3, 2)
    @test size(v.b) == (2,)
    
    @test size(v.E) == (2, 4)
    @test size(v.Q) == (2, 4)
    @test size(v.R) == (2, 4)
    @test size(v.T) == (2, 7)
    @test size(v.U) == (2, 7)

    @test size(v.T_res) == (2, 7)
    @test size(v.U_res) == (2, 7)

    @test size(v.Λ) == (2, 4)
    @test size(v.Σ) == (2, 4)
    @test size(v.Ω) == (2, 4)
    @test size(v.Θ) == (2, 7)
    @test size(v.Ξ) == (2, 7)
end

@testset "check losses" begin
    @testset "check obj loss" begin
        model, v = init_dummy_model_and_var()
        @test obj_loss(model, v) > 0.0
    end

    @testset "check lagrangian loss" begin
        model, v = init_dummy_model_and_var()
        @test lagrangian_loss(model, v) > 0.0
    end
end

function get_plus_minus_evals(var_to_check, model::pdMISVMClassifier, v::misvm_vars)
    δ = .0001

    lower_bound = lagrangian_loss(model, v)
    var_to_check .+= δ
    calc_WmX_WyX!(v); calc_by!(v)
    plus_cost = lagrangian_loss(model, v)
    var_to_check .-= 2.0*δ
    calc_WmX_WyX!(v); calc_by!(v)
    minus_cost = lagrangian_loss(model, v)

    return lower_bound, minus_cost, plus_cost
end

@testset "check updates" begin
    @testset "check W update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        W_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.W, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check inexact/s update" begin
        model, v = init_dummy_model_and_var()
        l1 = inexact_loss(model, v)
        s_update!(model, v)
        l2 = inexact_loss(model, v)
        @test l2 < l1
        δ = .0001
        v.s .+= δ
        plus = inexact_loss(model, v)
        v.s .-= 2.0*δ
        minus = inexact_loss(model, v)
        @test l2 < minus
        @test l2 < plus
    end

    @testset "check b update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        b_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.b, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check E update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        E_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.E, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check Q update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        Q_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.Q, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check R update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        R_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.R, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check T update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        T_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.T, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check U update" begin
        model, v = init_dummy_model_and_var()
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
        model = pdMISVMClassifier(exact=true)
        X, y = init_dummy_data()
        misvm = machine(model, X, y)

        fit!(misvm, verbosity=1)
        pred = predict(misvm, X)

        @test pred == y
    end

    @testset "check inexact fit and predict" begin
        model = pdMISVMClassifier(exact=false)
        X, y = init_dummy_data()
        misvm = machine(model, X, y)

        fit!(misvm, verbosity=1)
        pred = predict(misvm, X)

        @test pred == y
    end
end
