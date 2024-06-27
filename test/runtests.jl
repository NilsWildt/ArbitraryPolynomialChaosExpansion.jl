import Pkg
using APCE
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


set_preferences!(APCE, "precompile_workload" => true; force=true)

DIT_PATH = joinpath(@__DIR__, "..", "..", "APCE.jl")
if isdir(DIT_PATH)
    Pkg.develop(; path=DIT_PATH)
else
    Pkg.add("APCE")
end

#############################################################################################
############################################################################################
@testset verbose = false showtiming = true "All tests" begin
    #############################################################################################
    @testset "aPC_OrthonormalBasis" begin
        @test aPC_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1) ≈ [1, -0.5, -2.0, 0.0, 1.5, -1.5, 0.0, 0.0, 4.5]
    end
    @testset "aPC_MultivariatePolynomialDegrees" begin
        @test APCE.aPC_MultivariatePolynomialDegrees(2, 1) == [0, 0, 1, 0, 1, 0]
        @test APCE.aPC_MultivariatePolynomialDegrees(2, 2) == [0, 0, 1, 0, 1, 2, 0, 1, 0, 2, 1, 0]
    end
    @testset "sort_two_arrays!" begin
        x = [3.0, 2.0, 1.0]
        y = [3.0, 2.0, 1.0]
        sort_two_arrays!(x, y)
        @test x == [1.0, 2.0, 3.0]
    end
    #############################################################################################

    @testset "Aqua.jl testset" begin
        Aqua.test_all(
            APCE;
            ambiguities=false,      # TODO: fix ambiguities
            unbound_args=true,     # TODO: fix unbound type parameters
            piracies=false,         # TODO: check the reported methods to be moved upstream
			deps_compat=false
        )
        @test length(Aqua.detect_unbound_args_recursively(APCE)) <= 16
    end
    #############################################################################################
    @testset "JET.jl testset" begin
        x = [3.0, 2.0, 1.0]
        y = [3.0, 2.0, 1.0]
        JET.@test_call target_modules = (@__MODULE__,) sort_two_arrays!(x, y) # should pass
    end

    begin
        mis = methodinstances(APCE)    # get all the compiled methodinstances for functions owned by the package
        # Now let's filter out the ones that pass without issue
        badmis = filter(mis) do mi
            !isempty(JET.get_reports(report_call(mi)))
            # JET.get_reports(report_call(mi))
        end
        @warn badmis
    end
    #############################################################################################
    @testset "PerfChecker.jl" begin
        @testset "Perf: aPCE_OrthonormalBasis" begin
            # Title of the alloc check (for logging purpose)
            title = "Perf: aPCE_OrthonormalBasis"

            # Dependencies needed to execute pre_alloc and alloc
            dependencies = [APCE]

            # Target of the alloc check
            targets = [APCE]

            X = rand(500, 2)

            # Code to trigger precompilation before the alloc check
            pre_alloc() = foreach(_ -> APCE.aPCE_OrthonormalBasis(X, 3), 1:10)

            # Code being allocations check
            alloc() = APCE.aPCE_OrthonormalBasis(X, 3)

            # Actual call to PerfChecker
            alloc_check(title, dependencies, targets, pre_alloc, alloc; path=@__DIR__, threads=10)
        end
        @testset "Perf: ForwardDiff aPCE_OrthonormalBasis" begin
            # Title of the alloc check (for logging purpose)
            title = "Perf: ForwardDiff aPCE_OrthonormalBasis"
            # Dependencies needed to execute pre_alloc and alloc
            dependencies = [APCE]
            # Target of the alloc check
            targets = [APCE]
            X = rand(500, 2)
            pre_alloc() = foreach(_ -> ForwardDiff.jacobian(X -> aPCE_OrthonormalBasis(X, 3), X), 1:10)
            # Code being allocations check
            alloc() = ForwardDiff.jacobian(X -> aPCE_OrthonormalBasis(X, 3), X)
            # Actual call to PerfChecker
            alloc_check(title, dependencies, targets, pre_alloc, alloc; path=@__DIR__, threads=10)
        end
    end
end