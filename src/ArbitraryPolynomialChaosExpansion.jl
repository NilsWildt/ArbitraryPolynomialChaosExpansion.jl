module ArbitraryPolynomialChaosExpansion
using TestItems
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!
using LinearAlgebra: LinearAlgebra, BLAS, transpose, cond
using LinearAlgebra: checksquare
using LazyArrays
using ReverseDiff: ReverseDiff
using LinearAlgebra: svd, norm, pinv, Diagonal, tr
using TypeUtils: as
using ChainRulesCore
using ErrorTypes

# Mooncake support via weak extension (MooncakeExt)
# On Julia 1.12+, Mooncake is not compatible due to compiler API changes
# The extension will only be loaded on compatible Julia versions when Mooncake is available
const MOONCAKE_AVAILABLE = @isdefined(Mooncake)

# Define CPU_MODEL safely with fallback
const CPU_MODEL = get(
    ENV, "CPU_MODEL", try
        Sys.cpu_info()[1].model
    catch
        ""
    end
)

# Use conditional loading directly. Use @debug, not @info, so importing the
# package does not print to the console on every `using` (library etiquette).
if Sys.isapple() && Sys.ARCH in (:aarch64, :arm64)
    @debug "Using `AppleAccelerate.jl` for Apple Silicon."
elseif Sys.ARCH == :x86_64 && occursin(r"intel"i, CPU_MODEL)
    @debug "Detected Intel x86_64 CPU. Loading `MKL.jl`."
    using MKL
end

# BLISBLAS is optional - only use if available (loaded via weak extension)
# On Apple Silicon, AppleAccelerate provides optimized BLAS already
const BLISBLAS_AVAILABLE = @isdefined(BLISBLAS)

import Optim: NewtonTrustRegion, Options, optimize, minimizer, minimum, LBFGS
import RegularizationTools: Lₖx₀, solve, RegularizationProblem, setupRegularizationProblem, to_general_form, to_standard_form, gcv_tr, gcv_svd, invert, Lₖ, NelderMead, LₖB, Lₖx₀B, LₖDₓ, Lₖx₀Dₓ, LₖDₓB, Lₖx₀DₓB
using CPUSummary: CPUSummary
using ChainRules: ChainRules
using ChainRulesCore: ChainRulesCore
using Combinatorics: Combinatorics, factorial
using DispatchDoctor: @stable
using Einsum: Einsum, @einsum
using Estrin: Estrin
using ForwardDiff: ForwardDiff, Dual
using LineSearches: LineSearches
using OnlineStats: OnlineStats, Extrema, Mean, Series, Variance, eachrow, value
using PolynomialRoots: PolynomialRoots
using PrecompileTools: @setup_workload, @compile_workload
using Random: Random, Xoshiro, shuffle
using RegularizationTools: RegularizationTools
using ReverseDiff: ReverseDiff
using StatsBase: StatsBase, fit!, mean, sum
using Zygote: Zygote, bufferfrom
using Suppressor: @suppress
using KernelAbstractions

BLAS.set_num_threads(CPUSummary.get_cpu_threads() ÷ 2)

export aPCE_FullBasis, aPCE_MultivariatePolynomialDegrees, aPCE_PsiPolynomialMatrix, aPCE_DerivativeBasis, compose_Ψ, GaussianCollocation, partitionTrainTest, special_sort_two_arrays!, train!, evaluate_Ψ, aPCE, aPCE_OrthonormalBasis, create_basis, create_centered_basis, CenteredBasis, create_recurrence_basis, RecurrenceCenteredBasis, normalization_functions, predict, UQ, PsiPolynomialMatrix_zygote, reverse_columns!, compute_moments!, MultiWaveletBasis, MultiWaveletElement, create_multiwavelet_basis

include("APCEfunctions.jl")
include("APCEhighlevel.jl")
include("APCEmultires.jl")
include("utils.jl")
include("APCEderivatives.jl")
include("APCEGradientOverrides.jl")

# include("TensorOperationsMooncakeExt.jl")
# using .TensorOperationsMooncakeExt

@setup_workload begin
    @suppress begin
        FT = Float64
        TrainingInput = rand(10, 2) |> Array{FT}
        TrainingOutput = rand(10, 2) |> Array{FT} |> Array{FT}
        @compile_workload begin
            degree = 1
            apc_instance = aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), is_orthonormal = true, center_data = true)
            train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_order = 2)
            predict(apc_instance, TrainingInput)
            UQ(apc_instance)
        end
    end
end

@testitem "ArbitraryPolynomialChaosExpansion_module_test" begin
    # Test basic module functionality
    @test isdefined(ArbitraryPolynomialChaosExpansion, :aPCE)
    @test isdefined(ArbitraryPolynomialChaosExpansion, :train!)
    @test isdefined(ArbitraryPolynomialChaosExpansion, :predict)
    @test isdefined(ArbitraryPolynomialChaosExpansion, :UQ)
    @test isdefined(ArbitraryPolynomialChaosExpansion, :GaussianCollocation)
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
        is_orthonormal = true,
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


@testitem "APCE_type_stability_test" begin
    # Test type stability of full workflow
    FT = Float64
    TrainingInput = rand(10, 2) |> Array{FT}
    TrainingOutput = rand(10, 2) |> Array{FT}

    # Test type stability of constructor with different options
    @inferred aPCE(TrainingInput, 1)
    # `is_orthonormal = true` is the default, so it is omitted here: the default
    # path is `@constprop`-specialized and `@inferred`-stable, whereas an
    # *explicit* `is_orthonormal` keyword goes through Julia's keyword sorter
    # (which `@constprop` cannot reach) and would infer a backend `Union`.
    @inferred aPCE(TrainingInput, 1; outdim = 2, center_data = true)

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
    normalize, inverse_normalize = ArbitraryPolynomialChaosExpansion.normalization_functions(x)
    @inferred ArbitraryPolynomialChaosExpansion.normalization_functions(x)
    @test typeof(normalize(x)) == typeof(x)
    @test typeof(inverse_normalize(x)) == typeof(x)

    # Test type stability of compute_Psi_element
    TrainingInput = rand(10, 2)
    MultivariatePolynomialDegrees = [0 0; 0 1; 1 0]
    OrthonormalBasis = rand(3, 3, 2)
    @inferred ArbitraryPolynomialChaosExpansion.compute_Psi_element(1, 1, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, 2)

    # Test type stability of evalpoly_two
    x = 2.0
    coeffs = [1.0, 2.0, 3.0]
    @inferred ArbitraryPolynomialChaosExpansion.evalpoly_two(x, coeffs)
    @test typeof(ArbitraryPolynomialChaosExpansion.evalpoly_two(x, coeffs)) == Float64

    # Test type stability of evaluate_derivative_horner
    @inferred ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(x, coeffs)
    @test typeof(ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(x, coeffs)) == Float64

    # Test type stability of evaluate_polynomial_horner_array
    x_array = [1.0, 2.0, 3.0]
    @inferred ArbitraryPolynomialChaosExpansion.evaluate_polynomial_horner_array(x_array, coeffs)
    @test typeof(ArbitraryPolynomialChaosExpansion.evaluate_polynomial_horner_array(x_array, coeffs)) == Vector{Float64}

    # Test type stability of train
    Ψ = rand(10, 5)
    y_rhs = rand(10, 2)
    @inferred ArbitraryPolynomialChaosExpansion.train(Ψ, y_rhs)
    @test typeof(ArbitraryPolynomialChaosExpansion.train(Ψ, y_rhs)) == Matrix{Float64}

    # Test type stability of aPCE_FullBasis
    Data = rand(10)
    Degree = 2
    @inferred ArbitraryPolynomialChaosExpansion.aPCE_FullBasis(Data, Degree)
    @test typeof(ArbitraryPolynomialChaosExpansion.aPCE_FullBasis(Data, Degree)) == Matrix{Float64}

    # Test type stability of GaussianCollocation
    input_dimensions = 2
    ExpansionDegree = 2
    # Create a proper orthonormal basis for testing
    Data = rand(100, input_dimensions)
    OrthonormalBasis = ArbitraryPolynomialChaosExpansion.create_basis(Data, ExpansionDegree)
    InputDistribution = rand(10, input_dimensions)
    NumberOfTerms = 6
    @inferred ArbitraryPolynomialChaosExpansion.GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis, InputDistribution, NumberOfTerms)
    @test typeof(ArbitraryPolynomialChaosExpansion.GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis, InputDistribution, NumberOfTerms)) == Matrix{Float64}
end

end
