# Copyright (c) 2024 wildt
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
using Einsum
using DispatchDoctor: @stable
using LinearAlgebra: qr, svd, pinv, I

export aPCE, predict_from_coeffs
mutable struct aPCE{T<:Real}
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

    @stable function aPCE(
        InputDistribution::AbstractVecOrMat{T},
        ExpansionDegree::Int64;
        outdim::Int64=1,
        is_orthonormal::Bool=true,
        s_marginals=1.0,
        s_interactions=1.0,
        center_data=true,
        do_gauss=false,
        kwargs...
    ) where {T}
        input_dimensions = Int64(size(InputDistribution, 2))
        gauss_one_order_more = 0
        if do_gauss
            gauss_one_order_more = 1
        end
        MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(input_dimensions, ExpansionDegree + gauss_one_order_more, s_marginals, s_interactions)
        NumberOfTerms = min(size(MultivariatePolynomialDegrees, 1), numberPolynomials(ExpansionDegree + gauss_one_order_more, input_dimensions))
        OrthonormalBasis = create_basis(InputDistribution, ExpansionDegree + gauss_one_order_more, Val(is_orthonormal); center_data=center_data)
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


@stable function UQ(apc::aPCE{T}; axis=1) where {T<:Real}
    # @info "=> aPCE Toolbox: UQ Arbitrary Polynomial Chaos ..."
    # @info "Computing the mean and variance of the output for dimension $axis"
    lc = Array{T}(apc.ExpansionCoefficients[:, axis])
    OutputMean = @views lc[1, :]
    OutputVar = @views sum(lc[2:end, :] .^ 2; dims=1)[:]
    return (OutputMean=OutputMean, OutputVar=OutputVar)
end

# function aPCE_PsiPolynomialMatrix(aPCE::aPCE{T}, TrainingInput::Array{S})::Array{T} where {T<:Real,S<:Real}
#     Psi = aPCE_PsiPolynomialMatrix(TrainingInput, aPCE.MultivariatePolynomialDegrees, aPCE.OrthonormalBasis)
#     return Psi
# end

# function aPCE_PsiPolynomialMatrix(aPCE::aPCE{T}, TrainingInput)::T where {T<:ForwardDiff.Dual}
#     Psi = aPCE_PsiPolynomialMatrix(TrainingInput, aPCE.MultivariatePolynomialDegrees, aPCE.OrthonormalBasis)
#     return Psi
# end

function aPCE_PsiPolynomialMatrix(aPCE::aPCE{T}, TrainingInput::S)::S where {T<:Real,S<:AbstractArray}
    # @info "" aPCE typeof(TrainingInput) typeof(aPCE)
    Psi = aPCE_PsiPolynomialMatrix(TrainingInput, aPCE.MultivariatePolynomialDegrees, aPCE.OrthonormalBasis)
    return Psi
end


function GaussianCollocation(aPCE::aPCE{T}, len=0; strategy=:PCM)::Matrix{T} where {T<:Real}
    @assert aPCE.do_gauss "Gaussian collocation requires the do_gauss flag to be set to true"

    # Generate all possible combinations of polynomial points
    degree = aPCE.ExpansionDegree
    num_dims = aPCE.input_dimensions
    point_indices = collect(1:(degree+1))

    # Create all combinations of point indices across dimensions
    point_combinations = stack(reduce(vcat, Iterators.product([point_indices for _ in 1:num_dims]...)))'

    # Sort combinations by sum of indices (lower total degree first)
    sorted_indices = sortperm(sum(point_combinations; dims=2); dims=1)
    sorted_combinations = point_combinations[sorted_indices[:], :]

    # Handle different collocation strategies
    if strategy == :FT
        # Full Tensor strategy - just return the index combinations
        result = sorted_combinations
    elseif strategy == :PCM
        # Probabilistic Collocation Method - map indices to actual polynomial roots

        # Calculate polynomial roots for each dimension
        polynomial_roots = zeros(degree + 1, num_dims)
        @inbounds for dim in 1:num_dims
            polynomial_basis = @views aPCE.OrthonormalBasis[:, :, dim]
            # Extract roots of the polynomial (every other entry from real part)
            polynomial_roots[:, dim] = reinterpret(
                T,
                PolynomialRoots.roots(polynomial_basis[degree+2, :])
            )[1:2:(end-1)]
        end

        # Sort roots by distance to distribution mean in each dimension
        mean_distances = abs.(polynomial_roots .- StatsBase.mean(aPCE.InputDistribution; dims=1)[:, :][1])
        sort_indices_by_dim = mapslices(sortperm, mean_distances, dims=1)

        @inbounds for dim in 1:num_dims
            polynomial_roots[:, dim] = polynomial_roots[sort_indices_by_dim[:, dim], dim]
        end

        # Map the sorted index combinations to actual collocation points
        collocation_points = zeros(size(sorted_combinations, 1), num_dims)

        @inbounds for row in axes(collocation_points, 1)
            for dim in axes(collocation_points, 2)
                idx = Int(sorted_combinations[row, dim])
                collocation_points[row, dim] = polynomial_roots[idx, dim]
            end
        end

        # Sort points by first dimension value
        collocation_points = sortslices(collocation_points, dims=1, by=x -> x[1])
        result = collocation_points
    else
        error("Unknown strategy: $strategy. Use :FT or :PCM.")
    end

    # Apply length constraint if provided
    if len != 0
        return result[1:min(len, size(result, 1)), :]
    else
        return result
    end
end


# @stable function train!(aPCE, TrainingInput, y_rhs; bayesian_inversion = :true, reg_order = 1)
#     @info "=> aPCE Toolbox: Training Arbitrary Polynomial Chaos ..."
#     T = eltype(TrainingInput)
#     # @info aPCE
#     if size(y_rhs, 2) == 1
#         y_rhs = reshape(y_rhs, :, 1)
#     end
#     # y_rhs = reduce(hcat, TrainingOutput)'
#     if aPCE.output_dimensions != size(y_rhs, 2)
#         # @warn "Output dimensions of the aPCE model and the training output do not match"
#         aPCE.output_dimensions = size(y_rhs, 2)
#         aPCE.ExpansionCoefficients = zeros(T, aPCE.NumberOfTerms, aPCE.output_dimensions)
#     end
#     # NumberOfTerms, InputDimensions = size(aPCE.MultivariatePolynomialDegrees)
#     # NCpoints = size(TrainingInput, 1)
#     # Psi = SMatrix{NumberOfTerms,NCpoints}(aPCE_PsiPolynomialMatrix(aPCE, TrainingInput)')
#     Psi = aPCE_PsiPolynomialMatrix(aPCE, TrainingInput)' |> Matrix{T}
#     # @warn "SPYING"
#     # display(UnicodePlots.spy(sparse(Psi)))
#     # @debug "" size(TrainingInput) size(TrainingOutput) size(Psi) typeof(Psi) typeof(TrainingOutput) typeof(TrainingInput) size(aPCE.ExpansionCoefficients) typeof(aPCE.ExpansionCoefficients)
#     # Psi_inv = pinv(Psi;rtol= sqrt(eps(real(float(oneunit(eltype(Psi)))))) )


#     Psi_inv = pinv(Psi, rtol = sqrt(eps(real(float(oneunit(eltype(Psi)))))))
#     @tensor aPCE.ExpansionCoefficients[i, k] = Psi_inv[i, j] * y_rhs[j, k]
#     # aPCE.ExpansionCoefficients = outer_product_kernel(cu(Psi_inv), cu(y_rhs))

#     if bayesian_inversion
#         @info "Using bayesian regularization y_rhs find the expansion coefficients"
#         x₀ = aPCE.ExpansionCoefficients # Quite a good first guess :) And pinv is quite stable.

#         for i in axes(y_rhs, 2)
#             @info "Bayesian regularization for axis $i"
#             aPCE.ExpansionCoefficients[:, i] .= invert(Psi, y_rhs[:, i], Lₖx₀(reg_order, view(x₀, :, i)); alg = :gcv_svd, method = LBFGS(linesearch = LineSearches.BackTracking()))
#         end
#     end
#     for k in axes(aPCE.ExpansionCoefficients, 2)
#         res = (@views sqrt(mean((Psi * aPCE.ExpansionCoefficients[:, k] .- y_rhs[:, k]) .^ 2)))
#         @info "Error for axis $k" res
#     end
#     return nothing
# end

function train!(aPCE, TrainingInput, y_rhs; bayesian_inversion=:true, reg_order=1)
    @info "=> aPCE Toolbox: Training Arbitrary Polynomial Chaos ..."
    T = eltype(TrainingInput)

    # Format the output data
    if size(y_rhs, 2) == 1
        y_rhs = reshape(y_rhs, :, 1)
    end

    # Update dimensions if needed
    if aPCE.output_dimensions != size(y_rhs, 2)
        aPCE.output_dimensions = size(y_rhs, 2)
        aPCE.ExpansionCoefficients = zeros(T, aPCE.NumberOfTerms, aPCE.output_dimensions)
    end

    # Compute the polynomial matrix
    Psi = aPCE_PsiPolynomialMatrix(aPCE, TrainingInput)' |> Matrix{T}

    # More robust pseudoinverse calculation
    try
        # Try the standard pinv with the specified tolerance
        Psi_inv = LinearAlgebra.pinv(Psi, rtol=sqrt(eps(real(float(oneunit(eltype(Psi)))))))
        @tensor aPCE.ExpansionCoefficients[i, k] = Psi_inv[i, j] * y_rhs[j, k]
    catch e
        @warn "Standard pinv failed, trying alternative approach" exception = e

        # Try direct solving with Tikhonov regularization
        λ = 1.0e-6  # Regularization parameter
        num_terms = size(aPCE.ExpansionCoefficients, 1)

        # Solve directly (Psi'*Psi + λ*I)*c = Psi'*y for each output dimension
        for k in axes(y_rhs, 2)
            try
                aPCE.ExpansionCoefficients[:, k] = (Psi' * Psi + λ * LinearAlgebra.I(num_terms)) \ (Psi' * y_rhs[:, k])
            catch e2
                @warn "Tikhonov regularization failed for output $k, trying SVD approach" exception = e2
                # Use SVD with more careful handling
                try
                    U, S, V = LinearAlgebra.svd(Psi)
                    # Threshold small singular values
                    tol = maximum(size(Psi)) * maximum(S) * eps(T)
                    S_inv = map(s -> s > tol ? 1 / s : zero(T), S)

                    # Compute pseudoinverse via SVD
                    Psi_inv_svd = V * Diagonal(S_inv) * U'
                    aPCE.ExpansionCoefficients[:, k] = Psi_inv_svd * y_rhs[:, k]
                catch e3
                    @error "All numerical approaches failed for output $k" exception = e3
                    # Last resort - try QR factorization for this output
                    F = LinearAlgebra.qr(Psi)
                    aPCE.ExpansionCoefficients[:, k] = F \ y_rhs[:, k]
                end
            end
        end
    end

    # Continue with bayesian inversion if enabled
    if bayesian_inversion
        @info "Using bayesian regularization to find the expansion coefficients"
        x₀ = copy(aPCE.ExpansionCoefficients)

        for i in axes(y_rhs, 2)
            @info "Bayesian regularization for axis $i"
            aPCE.ExpansionCoefficients[:, i] .= invert(
                Psi, y_rhs[:, i], Lₖx₀(reg_order, view(x₀, :, i));
                alg=:gcv_svd,
                method=LBFGS(linesearch=LineSearches.BackTracking())
            )
        end
    end

    # Compute and report errors
    for k in axes(aPCE.ExpansionCoefficients, 2)
        res = (@views sqrt(mean((Psi * aPCE.ExpansionCoefficients[:, k] .- y_rhs[:, k]) .^ 2)))
        @info "Error for axis $k" res
    end
    return nothing
end


@stable function predict(aPCE::aPCE{T}, PredictionInput)::Matrix{T} where {T<:Real}
    # @info "=> aPCE Toolbox: Prediction using Arbitrary Polynomial Chaos ..."
    Psi = aPCE_PsiPolynomialMatrix(aPCE, PredictionInput)
    TensorOperations.@tensor PredictionOutput[k, j] := Psi[i, k] * aPCE.ExpansionCoefficients[i, j]
    # PredictionOutput = outer_product_kernel(cu(Psi), cu(aPCE.ExpansionCoefficients))
    return PredictionOutput
end


@stable function predict_from_coeffs(aPCE::aPCE{T}, PredictionInput, θ) where {T<:ForwardDiff.Dual}
    Psi = aPCE_PsiPolynomialMatrix(aPCE, PredictionInput)
    @einsum PredictionOutput[k, j] := Psi[i, k] * θ[i, j]
    # PredictionOutput = outer_product_kernel(cu(Psi), cu(aPCE.ExpansionCoefficients))
    return PredictionOutput
end

@testitem "aPCE_constructor_test" begin
    # Test basic constructor
    TrainingInput = rand(10, 2)
    degree = 1
    apc = aPCE(TrainingInput, degree)
    @test apc.input_dimensions == 2
    @test apc.ExpansionDegree == 1
    @test apc.is_orthonormal == true
    @test apc.do_gauss == false

    # Test with different options
    apc2 = aPCE(TrainingInput, degree; outdim=3, is_orthonormal=false, do_gauss=true)
    @test apc2.output_dimensions == 3
    @test apc2.is_orthonormal == false
    @test apc2.do_gauss == true
end

@testitem "aPCE_predict_test" begin
    # Create test data
    TrainingInput = rand(10, 2)
    TrainingOutput = rand(10, 1)
    degree = 1
    apc = aPCE(TrainingInput, degree)

    # Train the model
    train!(apc, TrainingInput, TrainingOutput)

    # Test prediction
    test_input = rand(5, 2)
    prediction = predict(apc, test_input)
    @test size(prediction) == (5, 1)
    @test all(!isnan, prediction)
    @test all(!isinf, prediction)
end

@testitem "aPCE_UQ_test" begin
    # Create test data
    TrainingInput = rand(10, 2)
    TrainingOutput = rand(10, 1)
    degree = 1
    apc = aPCE(TrainingInput, degree)

    # Train the model
    train!(apc, TrainingInput, TrainingOutput)

    # Test UQ
    uq_result = UQ(apc)
    @test haskey(uq_result, :OutputMean)
    @test haskey(uq_result, :OutputVar)
    @test length(uq_result.OutputMean) == 1
    @test length(uq_result.OutputVar) == 1
    @test all(!isnan, uq_result.OutputMean)
    @test all(!isnan, uq_result.OutputVar)
    @test all(!isinf, uq_result.OutputMean)
    @test all(!isinf, uq_result.OutputVar)
end

@testitem "aPCE_GaussianCollocation_test" begin
    # Create test data
    TrainingInput = rand(10, 2)
    degree = 1
    apc = aPCE(TrainingInput, degree; do_gauss=true)

    # Test Gaussian collocation
    collocation_points = GaussianCollocation(apc)
    @test size(collocation_points, 2) == 2
    @test all(!isnan, collocation_points)
    @test all(!isinf, collocation_points)

    # Test with different strategies
    collocation_points_ft = GaussianCollocation(apc; strategy=:FT)
    collocation_points_pcm = GaussianCollocation(apc; strategy=:PCM)
    @test size(collocation_points_ft, 2) == 2
    @test size(collocation_points_pcm, 2) == 2
end

@testitem "aPCE_type_stability_test" begin
    # Test type stability of constructor
    TrainingInput = rand(10, 2)
    degree = 1
    @inferred aPCE(TrainingInput, degree)
    @inferred aPCE(TrainingInput, degree; outdim=3, is_orthonormal=false, do_gauss=true)

    # Test type stability of predict
    apc = aPCE(TrainingInput, degree)
    TrainingOutput = rand(10, 1)
    train!(apc, TrainingInput, TrainingOutput)
    test_input = rand(5, 2)
    @inferred predict(apc, test_input)

    # Test type stability of UQ
    @inferred UQ(apc)

    # Test type stability of GaussianCollocation
    apc_gauss = aPCE(TrainingInput, degree; do_gauss=true)
    @inferred GaussianCollocation(apc_gauss)
    @inferred GaussianCollocation(apc_gauss; strategy=:FT)
    @inferred GaussianCollocation(apc_gauss; strategy=:PCM)
end
