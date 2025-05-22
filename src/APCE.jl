module APCE
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!
using LinearAlgebra: LinearAlgebra, BLAS, transpose
using LinearAlgebra: checksquare
using LazyArrays
using LinearAlgebra: svd, norm, pinv, Diagonal, tr
# Define CPU_MODEL safely with fallback
const CPU_MODEL = get(ENV, "CPU_MODEL", try
    Sys.cpu_info()[1].model
catch
    ""
end)

# Use conditional loading directly
if Sys.isapple() && Sys.ARCH in (:aarch64, :arm64)
    @info "Using `AppleAccelerate.jl` for Apple Silicon."
    using AppleAccelerate
    
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
# using ReverseDiff: ReverseDiff
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

export run, create_basis!, aPCE_FullBasis, aPCE_MultivariatePolynomialDegrees, aPCE_PsiPolynomialMatrix, compose_Ψ, GaussianCollocation, partitionTrainTest, special_sort_two_arrays!, train!, evaluate_Ψ, aPCE, aPCE_MultivariatePolynomialDegrees, aPCE_OrthonormalBasis, create_basis, GaussianCollocation, normalization_functions, partitionTrainTest, predict, train!, UQ, aPCE_PsiPolynomialMatrix_zygote!, aPCE_PsiPolynomialMatrix_zygote


include("APCEfunctions.jl")
include("APCEhighlevel.jl")
include("utils.jl")

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

end
