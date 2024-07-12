import Pkg
using Chairmarks
using DrWatson
using ForwardDiff
using JET
using MethodAnalysis
using PerfChecker
using Preferences
using Random
using Test
using Aqua
using Supposition
using Revise
using Suppressor
using BenchmarkTools

using APCE

set_preferences!(APCE, "precompile_workload" => true; force=true)

DIT_PATH = joinpath(@__DIR__, "..", "..", "APCE.jl")
if isdir(DIT_PATH)
    Pkg.develop(; path=DIT_PATH)
else
    Pkg.add("APCE")
end

############################################################################################
#############################classical tests################################################
@testset verbose = true showtiming = true "All tests" begin
    @testset "aPCE_OrthonormalBasis" begin
        @test APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true)) ≈ [1.0 0.0; -0.5 1.5]
        @test APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false)) ≈ [1.0 0.0; -0.5 1.5]
        @inferred APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false))
        @inferred APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true))
    end

    @testset "aPCE_MultivariatePolynomialDegrees" begin
        @test APCE.aPCE_MultivariatePolynomialDegrees(2, 1, 1.0, 1.0) == [0 0; 0 1; 1 0]
        @test APCE.aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0) == [0 0; 0 1; 1 0; 0 2; 1 1; 2 0]
        @inferred APCE.aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0)
    end

    # @testset "Property based testing" begin
    #     intgen = Data.Integers{UInt8}()
    #          # Define a property `foo` and feed it `Int8` from that generator
    #     #    @check max_examples= 10 function foo(i=intgen,j=intgen)
    #     #     @info "property test on $i and $j"
    #     #        MVPD = APCE.aPCE_MultivariatePolynomialDegrees(i, j, 1.0, 1.0)
    #     #        MVPD isa AbstractArray
    #     #    end
    # end

    ############################################################################################
    #############################Aqua################################################
    #############################################################################################
    # @testset "Aqua.jl testset" begin
    #     Aqua.test_all(
    #         APCE;
    #         ambiguities=true,      # TODO: fix ambiguities
    #         stale_deps=false,
    #         unbound_args=true,     # TODO: fix unbound type parameters
    #         piracies=false,         # TODO: check the reported methods to be moved upstream
    #         deps_compat=false
    #     )
    #     @test length(Aqua.detect_unbound_args_recursively(APCE)) <= 16
    # end
    #############################################################################################

    @testset "JET.jl testset" begin
        let
            FT = Float64
            TrainingInput = rand(10, 2) |> Array{FT}
            TrainingOutput = rand(10, 2) |> Array{FT} |> Array{FT}
            degree = 2
            apc_instance = aPCE(TrainingInput, degree; outdim=size(TrainingOutput, 2), OrthonormalRepresentation=true, normalize_data=true)
            JET.@test_opt target_modules = (@__MODULE__,) aPCE(TrainingInput, degree; outdim=size(TrainingOutput, 2), OrthonormalRepresentation=true, normalize_data=true)
            JET.@test_opt target_modules = (@__MODULE__,) train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion=true, reg_order=3)
            JET.@test_opt target_modules = (@__MODULE__,) predict(apc_instance, TrainingInput)
            JET.@test_opt target_modules = (@__MODULE__,) UQ(apc_instance)
        end
    end


    
    
    #################################Performance####################################################
    # @testset "PerfChecker.jl" begin
    #     @testset "Perf: aPCE_OrthonormalBasis allocs" begin
    #         # Title of the alloc check (for logging purpose)
    #         title = "Perf: aPCE_OrthonormalBasis allocs"
    #         # Dependencies needed to execute pre_alloc and alloc
    #         dependencies = [APCE]
    #         # Target of the alloc check
    #         targets = [APCE]
    #         X = rand(500, 2)
    #         # Code to trigger precompilation before the alloc check
    #         pre_alloc() = foreach(_ -> APCE.aPCE_OrthonormalBasis(X, 3), 1:2)
    #         # Code being allocations check
    #         alloc() = APCE.aPCE_OrthonormalBasis(X, 3)
    #         # Actual call to PerfChecker
    #         alloc_check(title, dependencies, targets, pre_alloc, alloc; path=@__DIR__, threads=10)
    #     end

    #     # @testset "Perf: aPCE_OrthonormalBasis speed" begin
    #     #     # Title of the alloc check (for logging purpose)
    #     #     title = "Perf: aPCE_OrthonormalBasis speed"
    #     #     # Dependencies needed to execute pre_alloc and alloc
    #     #     dependencies = [APCE]
    #     #     # Target of the alloc check
    #     #     targets = [APCE]
    #     #     X = rand(500, 2)
    #     #     # Code to trigger precompilation before the alloc check
    #     #     pre_alloc() = foreach(_ -> APCE.aPCE_OrthonormalBasis(X, 3), 1:2)
    #     #     # Code being allocations check
    #     #     bench = @be  APCE.aPCE_OrthonormalBasis(X, 3) evals = 10 samples = 10 seconds = 120
    #     #     store_benchmark(bench, target; path=@__DIR__)
    #     # end

    #     # @testset "Perf: ForwardDiff aPCE_OrthonormalBasis" begin
    #     #     # Title of the alloc check (for logging purpose)
    #     #     title = "Perf: ForwardDiff aPCE_OrthonormalBasis"
    #     #     # Dependencies needed to execute pre_alloc and alloc
    #     #     dependencies = [APCE]
    #     #     # Target of the alloc check
    #     #     targets = [APCE]
    #     #     X = rand(500, 2)
    #     #     pre_alloc() = foreach(_ -> ForwardDiff.jacobian(X -> aPCE_OrthonormalBasis(X, 3), X), 1:10)
    #     #     # Code being allocations check
    #     #     alloc() = ForwardDiff.jacobian(X -> aPCE_OrthonormalBasis(X, 3), X)
    #     #     # Actual call to PerfChecker
    #     #     alloc_check(title, dependencies, targets, pre_alloc, alloc; path=@__DIR__, threads=10)
    #     # end
    # end
    # alloc_plot([APCE])
    # bench_plot([APCE])
end