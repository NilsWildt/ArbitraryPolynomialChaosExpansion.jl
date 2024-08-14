module APCE
import Optim: NewtonTrustRegion, Options, optimize, minimizer, minimum, LBFGS, IPNewton
import RegularizationTools: Lₖx₀, solve, RegularizationProblem, setupRegularizationProblem, to_general_form, to_standard_form, gcv_tr, gcv_svd, invert, Lₖ, NelderMead, LₖB, Lₖx₀B, LₖDₓ, Lₖx₀Dₓ, LₖDₓB, Lₖx₀DₓB
using CPUSummary: CPUSummary
using ChainRules: ChainRules
using ChainRulesCore: ChainRulesCore
using Combinatorics: Combinatorics, factorial
using ComponentArrays: ComponentArrays
using DifferentiationInterface: DifferentiationInterface
using DispatchDoctor: @stable
using SparseArrays
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
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!
using LinearAlgebra: LinearAlgebra, BLAS, transpose
using LinearAlgebra: checksquare
using LazyArrays
using LinearAlgebra: svd, norm, pinv, Diagonal, tr
using Octavian: Octavian
using OnlineStats: OnlineStats, Extrema, Mean, Series, Variance, eachrow, value
using Polyester: Polyester, @batch
using PolynomialRoots: PolynomialRoots
using Polynomials: Polynomials, degree
using PrecompileTools: @setup_workload, @compile_workload    # this is a small dependency
using PrettyTables: PrettyTables
using Random: Random, Xoshiro, shuffle
using RegularizationTools: RegularizationTools
using ReverseDiff: ReverseDiff
using StaticArrays: StaticArrays
using StatsBase: StatsBase, fit!, mean, sum
import Bumper
using TensorOperations: TensorOperations, @tensoropt, @tensor, @butensor
using TimerOutputs: TimerOutputs
using Tracker: Tracker
using Tullio: Tullio, @tullio
using UnicodePlots: UnicodePlots
using UnrolledUtilities: UnrolledUtilities
using Zygote: Zygote, bufferfrom
using CUDA
using KernelAbstractions
using cuTENSOR
# CPUSummary.use_hwloc(true)

BLAS.set_num_threads(CPUSummary.get_cpu_threads() ÷ 2)

configdir(args...) = projectdir("configs", args...)
outputdir(args...) = projectdir("output", args...)

export run, normalization_functions, aPCE_MultivariatePolynomialDegrees, RowVecs, ColVecs, GaussianCollocation, train!, predict, UQ, partitionTrainTest, aPCE, aPCE_OrthonormalBasis, create_basis,
    evaluate_Ψ, train!

include("APCEfunctions.jl")
include("APCEderivatives.jl")
include("APCEhighlevel.jl")
include("utils.jl")

@setup_workload begin
    FT = Float64
    TrainingInput = rand(10, 2) |> Array{FT}
    TrainingOutput = rand(10, 2) |> Array{FT} |> Array{FT}
    @compile_workload begin
        degree = 2
        apc_instance = aPCE(TrainingInput, degree; outdim=size(TrainingOutput, 2), OrthonormalRepresentation=true, normalize_data=true)
        train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion=true, reg_order=3)
        predict(apc_instance, TrainingInput)
        UQ(apc_instance)
    end
end


end
