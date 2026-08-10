# Copyright (c) 2024 wildt
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
using Einsum
using DispatchDoctor: @stable
using LinearAlgebra: qr, svd, pinv, I

export aPCE, predict_from_coeffs

# Basis backend selection. The `basis` keyword is a `Val` (NOT a `Symbol`) on
# purpose: Julia specializes methods on keyword-argument *types*, so the choice
# flows into the returned `aPCE{T, B}` basis parameter and the constructor stays
# `@inferred`-clean — a `Symbol` keyword would make `Val(basis)` a runtime branch
# and infer a `Union`. Backends:
#   Val(:auto)        [default] recurrence when orthonormal, monomial/Vandermonde when not
#   Val(:monomial)             classic array-of-coefficients basis (`create_basis`)
#   Val(:recurrence)           three-term-recurrence basis (`create_recurrence_basis`;
#                             orthonormal-only, numerically stable to much higher degree)
#   Val(:multires)             multiresolution/multi-element basis (aMR-PC,
#                             `create_multiwavelet_basis`; orthonormal-only), split into
#                             two elements via the `split_dim`/`split_point` keywords
#                             forwarded from the constructor
#
# The default is `:auto` -> recurrence for the orthonormal case: benchmarked
# 2-5x faster than the monomial path (and leaner end-to-end) AND more robust on
# skewed marginals. `is_orthonormal = false` forces the monomial/Vandermonde
# path regardless, preserving that API.
#
# `basis` is a concrete `Val` (dispatched by method). `is_orthonormal` is a `Bool`
# and is branched *directly* (ternary), NOT wrapped in `Val`. Wrapping a literal
# Bool in `Val(::Bool)` infers to `Val` -- a `Val{true} ∪ Val{false}` union --
# which erases the literal and makes *every* `aPCE(...)` call infer a `Union`
# basis-parameter (`aPCE{T, Union{RecurrenceCenteredBasis, Array}}`). A direct
# ternary keeps the literal constant, so each call site that passes the flag
# literally infers a concrete `aPCE{T, B}` (enforced by the `@inferred aPCE(...)`
# tests); a runtime-Bool call site infers a union, which is an acceptable cost of
# carrying the backend in the type parameter. For `:monomial` both branches return
# `Array{T,3}` (concrete regardless of the flag); `:recurrence` is orthonormal by
# construction and rejects `is_orthonormal = false`.
_aPCE_build_basis(::Val{:auto}, is_orthonormal::Bool, x, degree, center_data) =
    is_orthonormal ? create_recurrence_basis(x, degree) :
    create_basis(x, degree, Val(false); center_data = center_data)
_aPCE_build_basis(::Val{:monomial}, is_orthonormal::Bool, x, degree, center_data) =
    create_basis(x, degree, Val(is_orthonormal); center_data = center_data)
_aPCE_build_basis(::Val{:recurrence}, is_orthonormal::Bool, x, degree, _center_data) =
    is_orthonormal ? create_recurrence_basis(x, degree) :
    throw(ArgumentError("basis = Val(:recurrence) requires is_orthonormal = true"))
_aPCE_build_basis(::Val{:multires}, is_orthonormal::Bool, x, degree, _center_data; kwargs...) =
    is_orthonormal ? create_multiwavelet_basis(x, degree; kwargs...) :
    throw(ArgumentError("basis = Val(:multires) requires is_orthonormal = true"))

# `aPCE{T, B}`: `B` is the basis-backend container type (an `AbstractArray{T}`
# for `:monomial`, a `RecurrenceCenteredBasis{T}` for `:recurrence`). Carrying
# it as a type parameter keeps field access — and therefore predict/UQ/train! —
# type-stable regardless of which backend is chosen.
mutable struct aPCE{T <: Real, B, A1 <: AbstractArray{T}, A2 <: AbstractArray{Int64}}
    const InputDistribution::A1 # in [ d x N-samples]
    const input_dimensions::Int64
    output_dimensions::Int64
    const ExpansionDegree::Int64
    const NumberOfTerms::Int64
    const MultivariatePolynomialDegrees::A2
    const is_orthonormal::Bool # if false: then vandermonde// full basis
    const OrthonormalBasis::B
    ExpansionCoefficients::Matrix{T}
    do_gauss::Bool

    # Trivial all-fields inner constructor. All build logic lives in the outer
    # `@constprop` constructor below: `Base.@constprop` does not apply to inner
    # constructors, so the keyword-value propagation needed for a concrete
    # `aPCE{T, B}` return type must live in a module-level method.
    function aPCE(
            InputDistribution::A1,
            input_dimensions::Int64,
            output_dimensions::Int64,
            ExpansionDegree::Int64,
            NumberOfTerms::Int64,
            MultivariatePolynomialDegrees::A2,
            is_orthonormal::Bool,
            OrthonormalBasis::B,
            ExpansionCoefficients::Matrix{T},
            do_gauss::Bool,
        ) where {T <: Real, B, A1 <: AbstractArray{T}, A2 <: AbstractArray{Int64}}
        return new{T, B, A1, A2}(
            InputDistribution,
            input_dimensions,
            output_dimensions,
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

# Outer constructor: the public entry point and all build logic.
#
# NOT `@stable`: the `basis`/`is_orthonormal` keywords make the returned
# `aPCE{T, B}` basis-parameter depend on the selected backend, so the generic
# signature is a small type union that DispatchDoctor cannot certify. Each
# concrete call site is still stable (enforced by the `@inferred aPCE(...)`
# tests), via the two mechanisms below.
#
# `Base.@constprop :aggressive` constant-propagates the `is_orthonormal` *keyword
# value* into the body. Keyword *types* always dispatch (that is how `basis::Val`
# selects the backend method), but keyword *values* are not propagated by default
# -- without `@constprop` the `:auto` ternary (recurrence vs monomial) would infer
# a `Union` and break `@inferred`. With it, a literal `is_orthonormal` folds to
# one branch -> concrete `aPCE{T, B}`. (`@constprop` does not work on inner
# constructors, which is why the logic lives here, not in the struct block.) Cost:
# a few extra specializations of a constructor called once per model -- negligible.
Base.@constprop :aggressive function aPCE(
        InputDistribution::AbstractVecOrMat{T},
        ExpansionDegree::Int64;
        outdim::Int64 = 1,
        is_orthonormal::Bool = true,
        s_marginals = 1.0,
        s_interactions = 1.0,
        center_data = true,
        do_gauss = false,
        basis::Val = Val(:auto),
        kwargs...,
    ) where {T}
    input_dimensions = Int64(size(InputDistribution, 2))
    gauss_one_order_more = 0
    if do_gauss
        gauss_one_order_more = 1
    end
    MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(input_dimensions, ExpansionDegree + gauss_one_order_more, s_marginals, s_interactions)
    NumberOfTerms = min(size(MultivariatePolynomialDegrees, 1), numberPolynomials(ExpansionDegree + gauss_one_order_more, input_dimensions))
    OrthonormalBasis = _aPCE_build_basis(basis, is_orthonormal, InputDistribution, ExpansionDegree + gauss_one_order_more, center_data; kwargs...)
    ExpansionCoefficients = zeros(T, NumberOfTerms, outdim)
    return aPCE(
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

import Base.show


@stable function show(io::IO, aPCE::aPCE{T, B, A1, A2}) where {T <: Real, B, A1, A2}
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


@stable function UQ(apc::aPCE{T, B, A1, A2}; axis = 1) where {T <: Real, B, A1, A2}
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

function aPCE_PsiPolynomialMatrix(aPCE::aPCE{T, B, A1, A2}, TrainingInput::S)::S where {T <: Real, B, A1, A2, S <: AbstractArray}
    # @info "" aPCE typeof(TrainingInput) typeof(aPCE)
    Psi = aPCE_PsiPolynomialMatrix(TrainingInput, aPCE.MultivariatePolynomialDegrees, aPCE.OrthonormalBasis)
    return Psi
end

"""
    aPCE_DerivativeBasis(
        apc::aPCE{T}, x::AbstractVector{S}, order::Integer,
    ) where {T, S} -> Matrix

Compute the `order`-th physical-space derivative of the PCE basis at the scalar
physical points `x`, returned as a `(NumberOfTerms, length(x))` matrix in the
same orientation as [`aPCE_PsiPolynomialMatrix`](@ref) (terms × points).

This is the analytic counterpart of nesting `ForwardDiff.derivative` on the Psi
matrix evaluation: the basis is differentiated through the three-term
recurrence (or monomial coefficients) instead of dual numbers, which is
~9.5× faster at `order = 1` and ~3.5× faster at `order = 4` on the
`RecurrenceCenteredBasis` path while staying exact to machine precision.

# Arguments
- `apc::aPCE{T}`: Expansion whose basis is differentiated. Its `OrthonormalBasis`
  backend selects the algorithm: `RecurrenceCenteredBasis` uses the analytic
  derivative recurrence; `CenteredBasis` / monomial coefficient arrays use
  `derivative_coeffs` + Horner evaluation; anything else raises an error.
- `x::AbstractVector{S}`: Physical points at which to evaluate the derivatives
  (a vector of scalar coordinates, i.e. a one-dimensional domain).
- `order::Integer`: Derivative order (≥ 0). `order = 0` reproduces
  `aPCE_PsiPolynomialMatrix(apc, reshape(x, :, 1))` exactly.

# Returns
- `Matrix`: Derivative basis of size `(NumberOfTerms, length(x))`, entry
  `[i, j] = ∂^order Ψ_i / ∂x^order` at `x[j]`.

# Details
The basis is standardized internally as `z = (x - μ)/σ` (the same per-dimension
statistics used by `aPCE_PsiPolynomialMatrix`), the derivative is taken in
`z`-space, and the chain rule `∂ᵏ/∂xᵏ = (1/σ)ᵏ ∂ᵏ/∂zᵏ` is applied so the
result is in physical units. Currently supports one input dimension;
multi-dimensional domains need a partial-derivative convention that is not yet
part of this API.

# Examples
```julia
x_ref = randn(10_000)
apc = aPCE(reshape(x_ref, :, 1), 6)
D1 = aPCE_DerivativeBasis(apc, [-1.0, 0.0, 1.0], 1)   # 7 × 3, ∂Ψ/∂x in physical units
```
"""
function aPCE_DerivativeBasis(
        apc::aPCE{T},
        x::AbstractVector{S},
        order::Integer,
    ) where {T <: Real, S <: Real}
    order >= 0 || throw(ArgumentError("derivative order must be non-negative"))
    apc.input_dimensions == 1 || throw(
        DimensionMismatch(
            "aPCE_DerivativeBasis currently supports one input dimension (1D " *
            "physical points); got $(apc.input_dimensions)",
        ),
    )
    return _aPCE_derivative_basis_backend(apc, x, Int(order), apc.OrthonormalBasis)
end

# Backend selection for `aPCE_DerivativeBasis`: the algorithm depends on how the
# basis stores its polynomials (recurrence coefficients vs monomial coefficients),
# so it is dispatched on the `OrthonormalBasis` field type — the same pattern as
# `aPCE_PsiPolynomialMatrix_zygote` / `_basis_nodes`.

function _aPCE_derivative_basis_backend(apc::aPCE, x::AbstractVector, order::Int, basis)
    throw(
        ArgumentError(
            "aPCE_DerivativeBasis does not support basis backend $(typeof(basis))",
        ),
    )
end

# RecurrenceCenteredBasis: analytic derivative recurrence in z-space, then the
# (1/σ)^order chain-rule scaling to physical units.
function _aPCE_derivative_basis_backend(
        apc::aPCE{T},
        x::AbstractVector{S},
        order::Int,
        rb::RecurrenceCenteredBasis{T},
    ) where {T <: Real, S <: Real}
    degs = apc.MultivariatePolynomialDegrees
    P = apc.NumberOfTerms
    n = length(x)

    # 1D standardization (input_dimensions == 1 asserted by the public wrapper).
    μ = rb.μ[1]
    σ = rb.σ[1]
    z = (x .- μ) ./ σ
    vals = _orthonormal_recurrence_derivative_order(
        z, @view(rb.α[:, 1]), @view(rb.β[:, 1]), rb.degree, order,
    )

    invσ = inv(σ)
    chain = invσ^order                 # ∂ᵏ/∂xᵏ = (1/σ)ᵏ ∂ᵏ/∂zᵏ
    out = Matrix{promote_type(T, S)}(undef, P, n)
    @inbounds for i in 1:P
        k = degs[i, 1] + 1
        for j in 1:n
            out[i, j] = vals[j, k] * chain
        end
    end
    return out
end

# CenteredBasis: z-space monomial coefficients, derivative_coeffs + Horner in
# z-space, then the same chain-rule scaling.
function _aPCE_derivative_basis_backend(
        apc::aPCE{T},
        x::AbstractVector{S},
        order::Int,
        cb::CenteredBasis,
    ) where {T <: Real, S <: Real}
    degs = apc.MultivariatePolynomialDegrees
    P = apc.NumberOfTerms
    n = length(x)

    μ = cb.μ[1]
    σ = cb.σ[1]
    z = (x .- μ) ./ σ
    invσ = inv(σ)
    chain = invσ^order

    out = Matrix{promote_type(T, S)}(undef, P, n)
    @inbounds for i in 1:P
        k = degs[i, 1] + 1
        coeffs = @views cb.basis[k, 1:k, 1]
        dcoeffs = coeffs
        for _ in 1:order
            dcoeffs = derivative_coeffs(dcoeffs)
        end
        for j in 1:n
            out[i, j] = evaluate_polynomial_horner_scalar(z[j], dcoeffs) * chain
        end
    end
    return out
end

# Monomial coefficient array (`create_basis`, e.g. the x-space full/orthonormal
# basis): coefficients are already in physical space, so no standardization and
# no chain-rule factor — just repeated derivative_coeffs + Horner at `x`.
function _aPCE_derivative_basis_backend(
        apc::aPCE{T},
        x::AbstractVector{S},
        order::Int,
        basis::AbstractArray,
    ) where {T <: Real, S <: Real}
    degs = apc.MultivariatePolynomialDegrees
    P = apc.NumberOfTerms
    n = length(x)

    out = Matrix{promote_type(T, S)}(undef, P, n)
    @inbounds for i in 1:P
        k = degs[i, 1] + 1
        coeffs = @views basis[k, 1:k, 1]
        dcoeffs = coeffs
        for _ in 1:order
            dcoeffs = derivative_coeffs(dcoeffs)
        end
        for j in 1:n
            out[i, j] = evaluate_polynomial_horner_scalar(x[j], dcoeffs)
        end
    end
    return out
end


function GaussianCollocation(aPCE::aPCE{T, B, A1, A2}, len = 0; strategy = :PCM)::Matrix{T} where {T <: Real, B, A1, A2}
    @assert aPCE.do_gauss "Gaussian collocation requires the do_gauss flag to be set to true"

    # Generate all possible combinations of polynomial points
    degree = aPCE.ExpansionDegree
    num_dims = aPCE.input_dimensions
    point_indices = collect(1:(degree + 1))

    # Create all combinations of point indices across dimensions
    point_combinations = stack(reduce(vcat, Iterators.product([point_indices for _ in 1:num_dims]...)))'

    # Sort combinations by sum of indices (lower total degree first)
    sorted_indices = sortperm(sum(point_combinations; dims = 2); dims = 1)
    sorted_combinations = point_combinations[sorted_indices[:], :]

    # Handle different collocation strategies
    if strategy == :FT
        # Full Tensor strategy - just return the index combinations
        result = sorted_combinations
    elseif strategy == :PCM
        # Probabilistic Collocation Method - map indices to actual polynomial roots

        # Calculate polynomial roots for each dimension. The nodes are the
        # roots of the top (degree + 1) orthonormal polynomial; `_basis_nodes`
        # dispatches on the basis backend (monomial → PolynomialRoots on the
        # coefficient vector; recurrence → Golub–Welsch eigenvalues of the
        # Jacobi matrix, which is the more stable route).
        polynomial_roots = zeros(T, degree + 1, num_dims)
        @inbounds for dim in 1:num_dims
            polynomial_roots[:, dim] = _basis_nodes(aPCE.OrthonormalBasis, dim, degree, T)
        end

        # Sort roots by distance to distribution mean in each dimension
        mean_distances = abs.(polynomial_roots .- StatsBase.mean(aPCE.InputDistribution; dims = 1)[:, :][1])
        sort_indices_by_dim = mapslices(sortperm, mean_distances, dims = 1)

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
        collocation_points = sortslices(collocation_points, dims = 1, by = x -> x[1])
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

"""
    _train_solve(Psi, y_rhs; bayesian_inversion = true, reg_order = 1)

Regularized least-squares solve shared by the global and per-element (multires)
`train!` methods. `Psi` is the (n_points × n_terms) design matrix, `y_rhs` is the
(n_points × outdim) training output; returns the (n_terms × outdim) coefficient
matrix.
"""
function _train_solve(
        Psi::AbstractMatrix{T},
        y_rhs::AbstractMatrix;
        bayesian_inversion = true,
        reg_order = 1,
    ) where {T <: Real}
    num_terms = size(Psi, 2)
    coeffs = Matrix{T}(undef, num_terms, size(y_rhs, 2))

    # Primary solve: Moore-Penrose pseudoinverse. `pinv` is SVD-based and does
    # not throw for finite input, so we test the result for finiteness instead
    # of catching an exception (keeps this path kernel/AD-safe). Only if the
    # result is non-finite do we drop into the per-output robust ladder
    # (Tikhonov normal equations -> SVD-thresholded pinv -> QR), each step
    # guarded by an explicit success/finiteness check rather than try/catch.
    Psi_inv = LinearAlgebra.pinv(Psi, rtol = sqrt(eps(real(float(oneunit(eltype(Psi)))))))
    # Replaced @tensor with explicit matrix multiplication for Mooncake AD compatibility
    primary = Psi_inv * y_rhs
    if all(isfinite, primary)
        coeffs .= primary
    else
        @warn "Standard pinv produced non-finite coefficients, using robust per-output fallback"
        λ = 1.0e-6  # Regularization parameter

        for k in axes(y_rhs, 2)
            # Tikhonov: (Psi'Psi + λI) c = Psi'y. The system matrix is symmetric
            # positive definite by construction, so factor with Cholesky and
            # check `issuccess` instead of catching a failure.
            G = LinearAlgebra.Symmetric(Psi' * Psi + λ * LinearAlgebra.I(num_terms))
            cF = LinearAlgebra.cholesky(G; check = false)
            if LinearAlgebra.issuccess(cF)
                tikhonov = cF \ (Psi' * y_rhs[:, k])
                if all(isfinite, tikhonov)
                    coeffs[:, k] = tikhonov
                    continue
                end
            end

            @warn "Tikhonov regularization failed for output $k, trying SVD approach"
            # SVD-thresholded pseudoinverse (small singular values dropped).
            U, S, V = LinearAlgebra.svd(Psi)
            tol = maximum(size(Psi)) * maximum(S) * eps(T)
            S_inv = map(s -> s > tol ? 1 / s : zero(T), S)
            svd_sol = (V * Diagonal(S_inv) * U') * y_rhs[:, k]
            if all(isfinite, svd_sol)
                coeffs[:, k] = svd_sol
                continue
            end

            @error "All numerical approaches failed for output $k, falling back to QR"
            # Last resort - QR factorization for this output.
            F = LinearAlgebra.qr(Psi)
            coeffs[:, k] = F \ y_rhs[:, k]
        end
    end

    # Continue with bayesian inversion if enabled
    if bayesian_inversion
        # @info "Using bayesian regularization to find the expansion coefficients"
        x₀ = copy(coeffs)

        # Validate initial guess - replace Inf/NaN with zeros
        if any(!isfinite, x₀)
            @warn "Initial coefficients contain Inf or NaN, using zero initial guess for regularization"
            x₀ .= zero(T)
        end

        for i in axes(y_rhs, 2)
            # @info "Bayesian regularization for axis $i"
            # Try the requested reg_order; fall back to lower orders if numerical issues arise.
            #
            # This try/catch is deliberately retained (it is the one exception to
            # the no-try/catch-in-compute rule): `invert` runs a third-party
            # GCV/LBFGS optimization (RegularizationTools + Optim) that can throw
            # deep inside on an ill-posed problem, and it exposes no success-flag
            # API to branch on. Catching at this third-party boundary is what
            # implements the order-fallback; on total failure we keep the pinv
            # solution already in `coeffs`. No autodiff runs through this path
            # (training is a fitting routine, not a differentiated kernel).
            for order in reg_order:-1:0
                try
                    coeffs[:, i] .= invert(
                        Psi, y_rhs[:, i], Lₖx₀(order, view(x₀, :, i));
                        alg = :gcv_svd,
                        method = LBFGS(linesearch = LineSearches.BackTracking())
                    )
                    if order < reg_order
                        @warn "Regularization order $reg_order failed for output $i, succeeded with order $order"
                    end
                    break
                catch e
                    if order > 0
                        continue
                    end
                    @warn "All regularization orders failed for output $i, keeping pinv solution" exception = e
                end
            end
        end
    end

    return coeffs
end

function train!(aPCE, TrainingInput, y_rhs; bayesian_inversion = true, reg_order = 1)
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

    aPCE.ExpansionCoefficients .= _train_solve(Psi, y_rhs; bayesian_inversion = bayesian_inversion, reg_order = reg_order)

    # Compute and report errors
    for k in axes(aPCE.ExpansionCoefficients, 2)
        res = (@views sqrt(mean((Psi * aPCE.ExpansionCoefficients[:, k] .- y_rhs[:, k]) .^ 2)))
        # @info "Error for axis $k" res
    end
    return nothing
end


@stable function predict(aPCE::aPCE{T, B, A1, A2}, PredictionInput) where {T <: Real, B, A1, A2}
    # @info "=> aPCE Toolbox: Prediction using Arbitrary Polynomial Chaos ..."
    Psi = aPCE_PsiPolynomialMatrix(aPCE, PredictionInput)
    # Replaced @tensor with explicit matrix multiplication for Mooncake AD compatibility
    PredictionOutput = Psi' * aPCE.ExpansionCoefficients
    return PredictionOutput
end


@stable function predict_from_coeffs(aPCE::aPCE{T, B, A1, A2}, PredictionInput, θ) where {T <: Real, B, A1, A2}
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
    apc2 = aPCE(TrainingInput, degree; outdim = 3, is_orthonormal = false, do_gauss = true)
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
    apc = aPCE(TrainingInput, degree; do_gauss = true)

    # Test Gaussian collocation
    collocation_points = GaussianCollocation(apc)
    @test size(collocation_points, 2) == 2
    @test all(!isnan, collocation_points)
    @test all(!isinf, collocation_points)

    # Test with different strategies
    collocation_points_ft = GaussianCollocation(apc; strategy = :FT)
    collocation_points_pcm = GaussianCollocation(apc; strategy = :PCM)
    @test size(collocation_points_ft, 2) == 2
    @test size(collocation_points_pcm, 2) == 2
end

@testitem "aPCE_type_stability_test" begin
    # Test type stability of constructor
    TrainingInput = rand(10, 2)
    degree = 1
    @inferred aPCE(TrainingInput, degree)
    # Name the backend explicitly (`basis = Val(:monomial)`) so the constructor
    # is `@inferred`-stable: `is_orthonormal = false` selects the Vandermonde
    # path, and an *explicit* `is_orthonormal` keyword routes through Julia's
    # keyword sorter (out of `@constprop`'s reach), which would infer a backend
    # `Union`. With `basis` named, the monomial return type is concrete.
    @inferred aPCE(TrainingInput, degree; outdim = 3, basis = Val(:monomial), is_orthonormal = false, do_gauss = true)

    # Test type stability of predict
    apc = aPCE(TrainingInput, degree)
    TrainingOutput = rand(10, 1)
    train!(apc, TrainingInput, TrainingOutput)
    test_input = rand(5, 2)
    @inferred predict(apc, test_input)

    # Test type stability of UQ
    @inferred UQ(apc)

    # Test type stability of GaussianCollocation
    apc_gauss = aPCE(TrainingInput, degree; do_gauss = true)
    @inferred GaussianCollocation(apc_gauss)
    @inferred GaussianCollocation(apc_gauss; strategy = :FT)
    @inferred GaussianCollocation(apc_gauss; strategy = :PCM)
end

@testitem "aPCE recurrence backend - workflow, type stability, basis invariance" begin
    using Random, LinearAlgebra, Statistics
    const APCE = ArbitraryPolynomialChaosExpansion

    rng = MersenneTwister(42)
    X = rand(rng, 60, 2)
    Y = reshape((@. sin(3X[:, 1]) + X[:, 2]^2), :, 1)
    Xtest = rand(rng, 6, 2)

    # The default `aPCE(...)` (no `basis` kwarg) resolves to the recurrence
    # backend when orthonormal — `Val(:auto)` -> `Val(:recurrence)`. Verify the
    # default actually produces a recurrence-backed model.
    @test aPCE(X, 6) isa aPCE{Float64, <:APCE.RecurrenceCenteredBasis, <:AbstractArray, <:AbstractArray}
    # `is_orthonormal = false` forces the monomial/Vandermonde path under :auto.
    @test aPCE(X, 2; is_orthonormal = false).OrthonormalBasis isa AbstractArray

    # `basis = Val(:recurrence)` selects the three-term-recurrence backend and
    # yields a distinct, fully-inferred concrete type parameter.
    apc_r = aPCE(X, 6; basis = Val(:recurrence))
    @test apc_r isa aPCE{Float64, <:APCE.RecurrenceCenteredBasis, <:AbstractArray, <:AbstractArray}
    @test apc_r.is_orthonormal == true
    io = IOBuffer(); show(io, apc_r); @test occursin("aPCE", String(take!(io)))

    apc_m = aPCE(X, 6; basis = Val(:monomial))
    train!(apc_m, X, Y; bayesian_inversion = false)
    train!(apc_r, X, Y; bayesian_inversion = false)

    # predict and UQ are basis-invariant: the recurrence and monomial backends
    # span the same orthonormal polynomial space (at degree > 4 the monomial path
    # is genuinely orthonormal too), so the fitted function and its moments agree.
    @test isapprox(predict(apc_m, Xtest), predict(apc_r, Xtest); atol = 1.0e-6)
    @test isapprox(UQ(apc_m).OutputMean, UQ(apc_r).OutputMean; atol = 1.0e-6)
    @test isapprox(UQ(apc_m).OutputVar, UQ(apc_r).OutputVar; atol = 1.0e-5)

    # Recurrence backend also works end-to-end at low degree (closed-form regime
    # for the monomial backend) and via Gaussian collocation (Golub–Welsch nodes).
    apc_lo = aPCE(X, 3; basis = Val(:recurrence))
    train!(apc_lo, X, Y; bayesian_inversion = false)
    @test all(isfinite, predict(apc_lo, Xtest))
    @test all(isfinite, UQ(apc_lo).OutputVar)

    apc_g = aPCE(X, 2; do_gauss = true, basis = Val(:recurrence))
    cp = GaussianCollocation(apc_g)
    @test size(cp, 2) == 2 && all(isfinite, cp)
    @test size(GaussianCollocation(apc_g; strategy = :FT), 2) == 2

    # Type stability across the recurrence path (concrete B in aPCE{T, B}).
    @test @inferred(aPCE(X, 3; basis = Val(:recurrence))) isa
        aPCE{Float64, <:APCE.RecurrenceCenteredBasis, <:AbstractArray, <:AbstractArray}
    @inferred predict(apc_r, Xtest)
    @inferred UQ(apc_r)
    @inferred GaussianCollocation(apc_g)

    # is_orthonormal = false with the recurrence backend is a contradiction.
    @test_throws ArgumentError aPCE(X, 3; basis = Val(:recurrence), is_orthonormal = false)
end
