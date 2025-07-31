module APCE
using TestItems
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!
using LinearAlgebra: LinearAlgebra, BLAS, transpose
using LinearAlgebra: checksquare
using LazyArrays
using ReverseDiff: ReverseDiff
using LinearAlgebra: svd, norm, pinv, Diagonal, tr
using Enzyme
using Mooncake: @from_rrule, DefaultCtx

# Define CPU_MODEL safely with fallback
const CPU_MODEL = get(
    ENV, "CPU_MODEL", try
        Sys.cpu_info()[1].model
    catch
        ""
    end
)

# Use conditional loading directly
if Sys.isapple() && Sys.ARCH in (:aarch64, :arm64)
    @info "Using `AppleAccelerate.jl` for Apple Silicon."
    # AppleAccelerate.@replaceBase sin cos tan
    # AppleAccelerate.@replaceBase asin acos atan
    # AppleAccelerate.@replaceBase sinh cosh tanh
    # AppleAccelerate.@replaceBase asinh acosh atanh
    # AppleAccelerate.@replaceBase exp exp2 expm1
    # AppleAccelerate.@replaceBase log log10 log2 log1p
    # AppleAccelerate.@replaceBase sqrt
    # AppleAccelerate.@replaceBase ceil floor trunc round
    # AppleAccelerate.@replaceBase abs
elseif Sys.ARCH == :x86_64 && occursin(r"intel"i, CPU_MODEL)
    @info "Detected Intel x86_64 CPU. Loading `MKL.jl`."
    using MKL
end


using BLISBLAS: BLISBLAS
import Optim: NewtonTrustRegion, Options, optimize, minimizer, minimum, LBFGS
import RegularizationTools: Lₖx₀, solve, RegularizationProblem, setupRegularizationProblem, to_general_form, to_standard_form, gcv_tr, gcv_svd, invert, Lₖ, NelderMead, LₖB, Lₖx₀B, LₖDₓ, Lₖx₀Dₓ, LₖDₓB, Lₖx₀DₓB
using CPUSummary: CPUSummary
using ChainRules: ChainRules
using ChainRulesCore: ChainRulesCore
using Combinatorics: Combinatorics, factorial
using ComponentArrays: ComponentArrays
using Estrin
# using DifferentiationInterface: DifferentiationInterface
using DispatchDoctor: @stable
# using SparseArrays
using DrWatson: DrWatson, projectdir
using Einsum: Einsum, @einsum
using Estrin: Estrin
using FastBroadcast: @.. # Unroll to speedup...
using ForwardDiff: ForwardDiff, Dual
# using InducingPoints: InducingPoints, CoverTree, RandomSubset, UniGrid, inducingpoints, kDPP
using Infiltrator: Infiltrator, @infiltrate
using Krylov: Krylov
using LazyGrids: LazyGrids
using LineSearches: LineSearches

# using Octavian: Octavian
using OnlineStats: OnlineStats, Extrema, Mean, Series, Variance, eachrow, value
using Polyester: Polyester, @batch
using PolynomialRoots: PolynomialRoots
using Polynomials: Polynomials, degree
using PrecompileTools: @setup_workload, @compile_workload    # this is a small dependency
using PrettyTables: PrettyTables
using Random: Random, Xoshiro, shuffle
using RegularizationTools: RegularizationTools
using ReverseDiff: ReverseDiff
# using StaticArrays: StaticArrays
using StatsBase: StatsBase, fit!, mean, sum
using Bumper
using TensorOperations: TensorOperations, @tensoropt, @tensor
using TimerOutputs: TimerOutputs
# using Tracker: Tracker
using UnicodePlots: UnicodePlots
using UnrolledUtilities: UnrolledUtilities
using Zygote: Zygote, bufferfrom
using Suppressor: @suppress
# using CUDA
using KernelAbstractions
# using cuTENSOR
# CPUSummary.use_hwloc(true)

BLAS.set_num_threads(CPUSummary.get_cpu_threads() ÷ 2)


configdir(args...) = projectdir("configs", args...)
outputdir(args...) = projectdir("output", args...)

export create_basis!, aPCE_FullBasis, aPCE_MultivariatePolynomialDegrees, aPCE_PsiPolynomialMatrix, compose_Ψ, GaussianCollocation, partitionTrainTest, special_sort_two_arrays!, train!, evaluate_Ψ, aPCE, aPCE_MultivariatePolynomialDegrees, aPCE_OrthonormalBasis, create_basis, GaussianCollocation, normalization_functions, partitionTrainTest, predict, train!, UQ, aPCE_PsiPolynomialMatrix_zygote, reverse_columns!, get_orthogonal_basis, compute_moments!

# Re-export the methods to ensure they're available
export aPCE_OrthonormalBasis

include("APCEfunctions.jl")
include("APCEhighlevel.jl")
include("utils.jl")
include("APCEderivatives.jl")

# include("TensorOperationsMooncakeExt.jl")
# using .TensorOperationsMooncakeExt

@setup_workload begin
    @suppress begin
        FT = Float64
        TrainingInput = rand(10, 2) |> Array{FT}
        TrainingOutput = rand(10, 2) |> Array{FT} |> Array{FT}
        @compile_workload begin
            degree = 1
            apc_instance = aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), OrthonormalRepresentation = true, center_data = true)
            train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_order = 2)
            predict(apc_instance, TrainingInput)
            UQ(apc_instance)
        end
    end
end

@testitem "APCE_module_test" begin
    # Test basic module functionality
    @test isdefined(APCE, :aPCE)
    @test isdefined(APCE, :train!)
    @test isdefined(APCE, :predict)
    @test isdefined(APCE, :UQ)
    @test isdefined(APCE, :GaussianCollocation)
end

@testitem "APCE_full_workflow_test" begin
    # Test complete workflow
    FT = Float64
    TrainingInput = rand(10, 2) |> Array{FT}
    TrainingOutput = rand(10, 2) |> Array{FT}

    # Create and train model
    degree = 1
    apc_instance = aPCE(
        TrainingInput, degree;
        outdim = size(TrainingOutput, 2),
        OrthonormalRepresentation = true,
        center_data = true
    )

    # Test training
    train!(
        apc_instance, TrainingInput, TrainingOutput;
        bayesian_inversion = true,
        reg_order = 2
    )

    # Test prediction
    prediction = predict(apc_instance, TrainingInput)
    @test size(prediction) == size(TrainingOutput)
    @test all(!isnan, prediction)
    @test all(!isinf, prediction)

    # Test UQ
    uq_result = UQ(apc_instance)
    @test haskey(uq_result, :OutputMean)
    @test haskey(uq_result, :OutputVar)
end

@testitem "APCE_config_test" begin
    # Test configuration functions
    @test isdefined(APCE, :configdir)
    @test isdefined(APCE, :outputdir)
end

@testitem "APCE_type_stability_test" begin
    # Test type stability of full workflow
    FT = Float64
    TrainingInput = rand(10, 2) |> Array{FT}
    TrainingOutput = rand(10, 2) |> Array{FT}

    # Test type stability of constructor with different options
    @inferred aPCE(TrainingInput, 1)
    @inferred aPCE(TrainingInput, 1; outdim = 2, OrthonormalRepresentation = true, center_data = true)

    # Create instance for further tests
    apc_instance = aPCE(TrainingInput, 1; outdim = 2)

    # Test type stability of training
    @inferred train!(apc_instance, TrainingInput, TrainingOutput)
    @inferred train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_order = 2)

    # Test type stability of prediction
    @inferred predict(apc_instance, TrainingInput)

    # Test type stability of UQ
    @inferred UQ(apc_instance)

end

@testitem "type_stability_tests" begin
    # Test type stability of normalization_functions
    x = rand(10, 2)
    normalize, inverse_normalize = APCE.normalization_functions(x)
    @inferred APCE.normalization_functions(x)
    @test typeof(normalize(x)) == typeof(x)
    @test typeof(inverse_normalize(x)) == typeof(x)

    # Test type stability of compute_Psi_element
    TrainingInput = rand(10, 2)
    MultivariatePolynomialDegrees = [0 0; 0 1; 1 0]
    OrthonormalBasis = rand(3, 3, 2)
    @inferred APCE.compute_Psi_element(1, 1, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, 2)

    # Test type stability of evalpoly_two
    x = 2.0
    coeffs = [1.0, 2.0, 3.0]
    @inferred APCE.evalpoly_two(x, coeffs)
    @test typeof(APCE.evalpoly_two(x, coeffs)) == Float64

    # Test type stability of evaluate_derivative_horner
    @inferred APCE.evaluate_derivative_horner(x, coeffs)
    @test typeof(APCE.evaluate_derivative_horner(x, coeffs)) == Float64

    # Test type stability of evaluate_polynomial_horner_array
    x_array = [1.0, 2.0, 3.0]
    @inferred APCE.evaluate_polynomial_horner_array(x_array, coeffs)
    @test typeof(APCE.evaluate_polynomial_horner_array(x_array, coeffs)) == Vector{Float64}

    # Test type stability of train
    Ψ = rand(10, 5)
    y_rhs = rand(10, 2)
    @inferred APCE.train(Ψ, y_rhs)
    @test typeof(APCE.train(Ψ, y_rhs)) == Matrix{Float64}

    # Test type stability of aPCE_FullBasis
    Data = rand(10)
    Degree = 2
    @inferred APCE.aPCE_FullBasis(Data, Degree)
    @test typeof(APCE.aPCE_FullBasis(Data, Degree)) == Matrix{Float64}

    # Test type stability of GaussianCollocation
    input_dimensions = 2
    ExpansionDegree = 2
    # Create a proper orthonormal basis for testing
    Data = rand(100, input_dimensions)
    OrthonormalBasis = APCE.create_basis(Data, ExpansionDegree)
    InputDistribution = rand(10, input_dimensions)
    NumberOfTerms = 6
    @inferred APCE.GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis, InputDistribution, NumberOfTerms)
    @test typeof(APCE.GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis, InputDistribution, NumberOfTerms)) == Matrix{Float64}
end

end
