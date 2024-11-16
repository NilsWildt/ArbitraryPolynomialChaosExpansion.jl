# Copyright (c) 2024 wildt
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
using Einsum
using DispatchDoctor: @stable
export aPCE, predict_from_coeffs
mutable struct aPCE{T <: Real}
    const InputDistribution::AbstractArray{T} # in [ d x N-samples]
    const input_dimensions::Int64
    output_dimensions::Int64
    const ExpansionDegree::Int64
    const NumberOfTerms::Int64
    const MultivariatePolynomialDegrees::AbstractArray{Int64}
    const is_orthonormal::Bool # if flase: then vandermonde// full basis
    const OrthonormalBasis::AbstractArray{T}
    ExpansionCoefficients::Matrix{T}
    do_gauss::Bool

    # Constructor
    @stable function aPCE(
            InputDistribution::AbstractVecOrMat{T},
            ExpansionDegree::Int64;
            outdim::Int64 = 1,
            is_orthonormal::Bool = true,
            s_marginals = 1.0,
            s_interactions = 1.0,
            center_data = true,
            do_gauss = false,
            kwargs...
        ) where {T}
        input_dimensions = Int64(size(InputDistribution, 2))
        gauss_one_order_more = 0
        if do_gauss
            gauss_one_order_more = 1
        end
        MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(input_dimensions, ExpansionDegree + gauss_one_order_more, s_marginals, s_interactions)
        NumberOfTerms = min(size(MultivariatePolynomialDegrees, 1), numberPolynomials(ExpansionDegree + gauss_one_order_more, input_dimensions))
        OrthonormalBasis = create_basis(InputDistribution, ExpansionDegree + gauss_one_order_more, Val(is_orthonormal); center_data = center_data)
        ExpansionCoefficients = zeros(T, NumberOfTerms, outdim)
        return new{T}(
            InputDistribution,
            input_dimensions,
            outdim,
            ExpansionDegree,
            NumberOfTerms,
            MultivariatePolynomialDegrees,
            is_orthonormal,
            OrthonormalBasis,
            ExpansionCoefficients,
            do_gauss
        )
    end
end

import Base.show


@stable function show(io::IO, aPCE::aPCE)
    println(io, "=> aPCE Toolbox: Prediction using Arbitrary Polynomial Chaos ...")
    println(io, "aPCE{$(typeof(aPCE).parameters[1])} Summary:")
    println(io, "Input Dimensions: ", aPCE.input_dimensions)
    println(io, "Output Dimensions: ", aPCE.output_dimensions)
    println(io, "Expansion Degree: ", aPCE.ExpansionDegree)
    println(io, "Number Of Terms: ", aPCE.NumberOfTerms)
    # Depending on the size, you might want to only show a preview of the arrays
    println(io, "Multivariate Polynomial Degrees: ", size(aPCE.MultivariatePolynomialDegrees))
    println(io, "Orthonormal Basis: Dimensions ", size(aPCE.OrthonormalBasis))
    println(io, "Expansion Coefficients: Length ", length(aPCE.ExpansionCoefficients))
    println(io, "do_gauss ", aPCE.do_gauss)
end


@stable function UQ(apc::aPCE{T}; axis = 1) where {T <: Real}
    # @info "=> aPCE Toolbox: UQ Arbitrary Polynomial Chaos ..."
    # @info "Computing the mean and variance of the output for dimension $axis"
    lc = Array{T}(apc.ExpansionCoefficients[:, axis])
    OutputMean = @views lc[1, :]
    OutputVar = @views sum(lc[2:end, :] .^ 2; dims = 1)[:]
    return (OutputMean = OutputMean, OutputVar = OutputVar)
end

# function aPCE_PsiPolynomialMatrix(aPCE::aPCE{T}, TrainingInput::Array{S})::Array{T} where {T<:Real,S<:Real}
#     Psi = aPCE_PsiPolynomialMatrix(TrainingInput, aPCE.MultivariatePolynomialDegrees, aPCE.OrthonormalBasis)
#     return Psi
# end

# function aPCE_PsiPolynomialMatrix(aPCE::aPCE{T}, TrainingInput)::T where {T<:ForwardDiff.Dual}
#     Psi = aPCE_PsiPolynomialMatrix(TrainingInput, aPCE.MultivariatePolynomialDegrees, aPCE.OrthonormalBasis)
#     return Psi
# end

function aPCE_PsiPolynomialMatrix(aPCE::aPCE{T}, TrainingInput::S)::S where {T <: Real, S <: AbstractArray}
    # @info "" aPCE typeof(TrainingInput) typeof(aPCE)
    Psi = aPCE_PsiPolynomialMatrix(TrainingInput, aPCE.MultivariatePolynomialDegrees, aPCE.OrthonormalBasis)
    return Psi
end

function GaussianCollocation(aPCE::aPCE{T}; strategy = :PCM) where {T <: Real}
    @assert aPCE.do_gauss "Gaussian collocation requires the do_gauss flag to be set to true"
    # @info aPCE
    PointsVector = 1:(aPCE.ExpansionDegree + 1) |> collect
    UniqueCombinations = stack(reduce(vcat, (Iterators.product([PointsVector for _ in 1:aPCE.input_dimensions]...))))'
    sort_indices = sortperm(sum(UniqueCombinations; dims = 2); dims = 1)
    SortUniqueCombinations = UniqueCombinations[sort_indices[:], :]
    if strategy == :FT
        TrainingInput = SortUniqueCombinations
        return Array(view(TrainingInput, :, (1:size(TrainingInput, 2))))
    elseif strategy == :PCM
        polynomial_roots = zeros(aPCE.input_dimensions, aPCE.ExpansionDegree + 1)
        @inbounds for d in Base.oneto(Int64(aPCE.input_dimensions))
            polynomial_basis = @views aPCE.OrthonormalBasis[:, :, d]
            @debug "" polynomial_basis
            polynomial_roots[d, :] = @view reinterpret(T, PolynomialRoots.roots(@views polynomial_basis[aPCE.ExpansionDegree + 2, :]))[1:2:(end - 1)]
        end
        temp = abs.(polynomial_roots .- StatsBase.mean(aPCE.InputDistribution; dims = 1)[:, :][1])
        temp_sort = mapslices(sortperm, temp, dims = 2)
        @inbounds for i in axes(polynomial_roots, 1)
            polynomial_roots[i, :] = @views polynomial_roots[i, temp_sort[i, :]]
        end
        collocation_points = zeros(aPCE.NumberOfTerms, aPCE.input_dimensions)
        @inbounds for i in 1:aPCE.NumberOfTerms
            for j in axes(SortUniqueCombinations, 2)
                collocation_points[i, j] = @views polynomial_roots[j, Int(SortUniqueCombinations[i, j])]
            end
        end
        collocation_points = sortslices(collocation_points, dims = 1, by = x -> x[1])
        return Array(view(collocation_points, :, (1:size(collocation_points, 2))))
    end
end

@stable function train!(aPCE, TrainingInput, y_rhs; bayesian_inversion = :true, reg_order = 3)
    @info "=> aPCE Toolbox: Training Arbitrary Polynomial Chaos ..."
    T = eltype(TrainingInput)
    # @info aPCE
    if size(y_rhs, 2) == 1
        y_rhs = reshape(y_rhs, :, 1)
    end
    # y_rhs = reduce(hcat, TrainingOutput)'
    if aPCE.output_dimensions != size(y_rhs, 2)
        # @warn "Output dimensions of the aPCE model and the training output do not match"
        aPCE.output_dimensions = size(y_rhs, 2)
        aPCE.ExpansionCoefficients = zeros(T, aPCE.NumberOfTerms, aPCE.output_dimensions)
    end
    # NumberOfTerms, InputDimensions = size(aPCE.MultivariatePolynomialDegrees)
    # NCpoints = size(TrainingInput, 1)
    # Psi = SMatrix{NumberOfTerms,NCpoints}(aPCE_PsiPolynomialMatrix(aPCE, TrainingInput)')
    Psi = aPCE_PsiPolynomialMatrix(aPCE, TrainingInput)' |> Matrix{T}
    # @warn "SPYING"
    # display(UnicodePlots.spy(sparse(Psi)))
    # @debug "" size(TrainingInput) size(TrainingOutput) size(Psi) typeof(Psi) typeof(TrainingOutput) typeof(TrainingInput) size(aPCE.ExpansionCoefficients) typeof(aPCE.ExpansionCoefficients)
    # Psi_inv = pinv(Psi;rtol= sqrt(eps(real(float(oneunit(eltype(Psi)))))) )


    Psi_inv = pinv(Psi; rtol = sqrt(eps(real(float(oneunit(eltype(Psi)))))))
    @tensor aPCE.ExpansionCoefficients[i, k] = Psi_inv[i, j] * y_rhs[j, k]
    # aPCE.ExpansionCoefficients = outer_product_kernel(cu(Psi_inv), cu(y_rhs))

    if bayesian_inversion
        @info "Using bayesian regularization y_rhs find the expansion coefficients"
        x₀ = aPCE.ExpansionCoefficients # Quite a good first guess :) And pinv is quite stable.

        for i in axes(y_rhs, 2)
            @info "Bayesian regularization for axis $i"
            aPCE.ExpansionCoefficients[:, i] .= invert(Psi, y_rhs[:, i], Lₖx₀(reg_order, view(x₀, :, i)); alg = :gcv_svd, method = LBFGS(linesearch = LineSearches.BackTracking()))
        end
    end
    for k in axes(aPCE.ExpansionCoefficients, 2)
        res = (@views sqrt(mean((Psi * aPCE.ExpansionCoefficients[:, k] .- y_rhs[:, k]) .^ 2)))
        @info "Error for axis $k" res
    end
    return nothing
end


@stable function predict(aPCE::aPCE{T}, PredictionInput)::Matrix{T} where {T <: Real}
    # @info "=> aPCE Toolbox: Prediction using Arbitrary Polynomial Chaos ..."
    Psi = aPCE_PsiPolynomialMatrix(aPCE, PredictionInput)
    TensorOperations.@tensor PredictionOutput[k, j] := Psi[i, k] * aPCE.ExpansionCoefficients[i, j]
    # PredictionOutput = outer_product_kernel(cu(Psi), cu(aPCE.ExpansionCoefficients))
    return PredictionOutput
end


@stable function predict_from_coeffs(aPCE::aPCE{T}, PredictionInput, θ) where {T <: ForwardDiff.Dual}
    Psi = aPCE_PsiPolynomialMatrix(aPCE, PredictionInput)
    @einsum PredictionOutput[k, j] := Psi[i, k] * θ[i, j]
    # PredictionOutput = outer_product_kernel(cu(Psi), cu(aPCE.ExpansionCoefficients))
    return PredictionOutput
end
