using Pkg: Pkg
using Chairmarks
using DrWatson
using ForwardDiff
# using JET
using MethodAnalysis
using Preferences
using Random
using Test
using Aqua
using Supposition
using Revise
using Suppressor
using BenchmarkTools

using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion

set_preferences!(ArbitraryPolynomialChaosExpansion, "precompile_workload" => true; force=true)

# DIT_PATH = joinpath(@__DIR__, "..", "..", "ArbitraryPolynomialChaosExpansion.jl")
# if isdir(DIT_PATH)
#     Pkg.develop(; path=DIT_PATH)
# else
#     Pkg.add("ArbitraryPolynomialChaosExpansion")
# end

############################################################################################
#############################classical tests################################################
@testset verbose = true showtiming = true "All tests" begin


    @testset "reverse_columns! Tests" begin
        mat1 = [1 2 3; 4 5 6; 7 8 9]
        expected1 = [3 2 1; 6 5 4; 9 8 7]
        APCE.reverse_columns!(mat1)
        @test mat1 == expected1

        mat2 = [1 2; 3 4; 5 6]
        expected2 = [2 1; 4 3; 6 5]
        APCE.reverse_columns!(mat2)
        @test mat2 == expected2

        mat3 = [1 2 3 4; 5 6 7 8]
        expected3 = [4 3 2 1; 8 7 6 5]
        APCE.reverse_columns!(mat3)
        @test mat3 == expected3

        @test typeof(APCE.reverse_columns!(mat1)) == Matrix{Int}
    end


    @testset "evaluate_derivative_horner Tests" begin
        @test APCE.evaluate_derivative_horner(2.0, [1.0, 2.0, 3.0]) == 14.0  # Derivative of 1 + 2x + 3x^2 at x=2
        @test APCE.evaluate_derivative_horner(0.0, [1.0, 2.0, 3.0]) == 2.0   # Derivative of 1 + 2x + 3x^2 at x=0
        @test APCE.evaluate_derivative_horner(1.0, [0.0, 0.0, 0.0]) == 0.0   # Derivative of 0 polynomial at x=1
        @test APCE.evaluate_derivative_horner(1.0, [5.0]) == 0.0             # Derivative of constant polynomial at x=1
        @test typeof(APCE.evaluate_derivative_horner(2.0, [1.0, 2.0, 3.0])) == Float64
        @inferred APCE.evaluate_derivative_horner(1.0, [5.0])
    end

    @testset "derivative_coeffs Tests" begin
        @test APCE.derivative_coeffs([1.0, 2.0, 3.0]) == [2.0, 6.0]  # Derivative of 1 + 2x + 3x^2
        @test APCE.derivative_coeffs([0.0, 0.0, 0.0]) == [0.0, 0.0]  # Derivative of 0 polynomial
        @test APCE.derivative_coeffs([5.0]) == [0.0]                 # Derivative of constant polynomial
        @test APCE.derivative_coeffs([1.0, -1.0, 1.0, -1.0]) == [-1.0, 2.0, -3.0]  # Derivative of 1 - x + x^2 - x^3
    end

    @testset "evalpoly_two Tests" begin
        @test APCE.evalpoly_two(2.0, [1.0, 2.0, 3.0, 4.0]) == 49.0  # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=2
        @test APCE.evalpoly_two(0.0, [1.0, 2.0, 3.0, 4.0]) == 1.0   # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=0
        @test APCE.evalpoly_two(1.0, [0.0, 0.0, 0.0, 0.0]) == 0.0   # Zero polynomial at x=1
        @test APCE.evalpoly_two(1.0, [5.0]) == 5.0                  # Constant polynomial at x=1
        @test APCE.evalpoly_two(2.0, [1.0, -1.0, 1.0, -1.0]) == -5.0  # Polynomial 1 - x + x^2 - x^3 at x=2
    end
    ###########################################################################################
    ############################Aqua################################################
    ############################################################################################
    @testset "Aqua.jl testset" begin
        Aqua.test_all(
            APCE;
            ambiguities=true,      # TODO: fix ambiguities
            stale_deps=false,
            unbound_args=true,     # TODO: fix unbound type parameters
            piracies=true,         # TODO: check the reported methods to be moved upstream
            deps_compat=false,
            project_extras=false,
            persistent_tasks=false,
        )
        @test length(Aqua.detect_unbound_args_recursively(APCE)) <= 16
    end
    ############################################################################################

    # @testset "JET.jl testset" begin
    #     let
    #         FT = Float64
    #         TrainingInput = rand(10, 2) |> Array{FT}
    #         TrainingOutput = rand(10, 2) |> Array{FT} |> Array{FT}
    #         degree = 2
    #         apc_instance = aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), is_orthonormal = true, normalize_data = true)
    #         JET.@test_opt target_modules = (@__MODULE__,) aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), is_orthonormal = true, normalize_data = true)
    #         JET.@test_opt target_modules = (@__MODULE__,) aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), is_orthonormal = false, normalize_data = true)
    #         JET.@test_opt target_modules = (@__MODULE__,) train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_order = 3)
    #         JET.@test_opt target_modules = (@__MODULE__,) predict(apc_instance, TrainingInput)
    #         JET.@test_opt target_modules = (@__MODULE__,) UQ(apc_instance)
    #     end
    # end

end
