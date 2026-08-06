# Copyright (c) 2024 Nils Wildt
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

#    include(srcdir("ArbitraryPolynomialChaosExpansion.jl"))
#     using .ArbitraryPolynomialChaosExpansion
# const APCE = ArbitraryPolynomialChaosExpansion

"""
Custom sorting element for simultaneous sorting of mean and variance
"""
struct SpecialCoSorterElement{T1, T2, T3}
    x::T1
    z::T2
    y::T3
end

"""
Custom sorter that sorts multiple arrays simultaneously based on the first array
"""
struct CoSorter{
        T1,
        T2,
        T3,
        A <: AbstractVecOrMat{T1},
        B <: AbstractVecOrMat{T2},
        C <: AbstractVecOrMat{T3},
    } <: AbstractVector{SpecialCoSorterElement{T1, T2, T3}}
    sortarray::A
    otherarray::B
    coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)
Base.getindex(c::CoSorter, i...) =
    SpecialCoSorterElement(
    getindex(c.sortarray, i...),
    getindex(c.otherarray, i...),
    getindex(c.coarray, i...),
)
Base.setindex!(c::CoSorter, t::SpecialCoSorterElement, i...) =
    (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c)

Base.isless(a::SpecialCoSorterElement, b::SpecialCoSorterElement) =
    isless(a.x, b.x) || (a.x == b.x && isless(a.z, b.z))
Base.Sort.defalg(v::C) where {T <: Union{Number, Missing}, C <: CoSorter{T}} =
    Base.DEFAULT_UNSTABLE

# ===== UTILITY FUNCTIONS =====

"""
    normalization_functions(matrix)

Create normalization and inverse normalization functions for a matrix.
Returns a tuple of (normalize_fn, inverse_normalize_fn).
"""
function normalization_functions(matrix)
    col_means = StatsBase.mean(matrix, dims = 1)
    col_stds = StatsBase.std(matrix, dims = 1)
    normalize = (x) -> (x .- col_means) ./ col_stds
    inverse_normalize = (x) -> x .* col_stds .+ col_means
    return normalize, inverse_normalize
end

"""
    special_sort_two_arrays!(x::AbstractArray, y::AbstractArray)

Sort array y based on the sorting order of the first two columns of array x.
"""
function special_sort_two_arrays!(x::AbstractArray, y::AbstractArray)
    T = CoSorter(x[:, 1], x[:, 2], y)
    sort!(T)
    x = T.sortarray
    return y = T.coarray
end

"""
    reverse_columns!(x)
Reverse the order of columns in each row of matrix x in-place.
"""
function reverse_columns!(x)
    if isempty(x)
        return x
    end
    @inbounds for row in axes(x, 1)
        x[row, :] = reverse(@view x[row, :])
    end
    return x
end

"""
    evalpoly_two(x, coeffs)

Evaluate polynomial with coefficients `coeffs` at point `x` using Horner's method with loop unrolling.
More efficient than standard `evalpoly` for performance-critical applications.
"""
function evalpoly_two(x, coeffs)
    i = lastindex(coeffs)
    out = coeffs[i]
    i -= 1
    fi = firstindex(coeffs)
    while i > fi
        out = muladd(out, x, coeffs[i])
        out = muladd(out, x, coeffs[i - 1])
        i -= 2
    end
    return i == fi ? muladd(out, x, @inbounds(coeffs[fi])) : out
end

# ReverseDiff-compatible dispatch
function evalpoly_two(x::ReverseDiff.TrackedReal, coeffs)
    isempty(coeffs) && throw(ArgumentError("coeffs cannot be empty"))
    i = lastindex(coeffs)
    out = coeffs[i]
    i -= 1
    fi = firstindex(coeffs)
    while i >= fi + 1
        out = out * x + coeffs[i]
        out = out * x + coeffs[i - 1]
        i -= 2
    end
    return i == fi ? out * x + coeffs[fi] : out
end

function evalpoly_two(x, coeffs::AbstractVector{<:ReverseDiff.TrackedReal})
    isempty(coeffs) && throw(ArgumentError("coeffs cannot be empty"))
    i = lastindex(coeffs)
    out = coeffs[i]
    i -= 1
    fi = firstindex(coeffs)
    while i >= fi + 1
        out = out * x + coeffs[i]
        out = out * x + coeffs[i - 1]
        i -= 2
    end
    return i == fi ? out * x + coeffs[fi] : out
end

function evalpoly_two(
        x::ReverseDiff.TrackedReal,
        coeffs::AbstractVector{<:ReverseDiff.TrackedReal},
    )
    isempty(coeffs) && throw(ArgumentError("coeffs cannot be empty"))
    i = lastindex(coeffs)
    out = coeffs[i]
    i -= 1
    fi = firstindex(coeffs)
    while i >= fi + 1
        out = out * x + coeffs[i]
        out = out * x + coeffs[i - 1]
        i -= 2
    end
    return i == fi ? out * x + coeffs[fi] : out
end


"""
    evaluate_polynomial_horner_array(x::AbstractVector, coeffs::AbstractVector)

Evaluate polynomial with coefficients `coeffs` at multiple points in vector `x`.
Returns a vector of evaluated values.
"""
@inline function evaluate_polynomial_horner_array(
        x::AbstractVector{T},
        coeffs::AbstractVector{S},
    ) where {T <: Real, S <: Real}
    R = promote_type(T, S)
    results = Vector{R}(undef, length(x))
    # Hoist the coefficient reversal out of the per-point loop: previously
    # `reverse(coeffs)` allocated a fresh reversed vector for every element of
    # `x` (O(length(x)) temporaries). Reversing once is allocation-equivalent to
    # a single temporary and leaves the numerical result unchanged.
    rcoeffs = reverse(coeffs)
    @inbounds for (i, xi) in enumerate(x)
        result = zero(R)
        @simd for coeff in rcoeffs
            result = muladd(result, xi, coeff)
        end
        results[i] = result
    end
    return results
end
"""
    evaluate_polynomial_horner_scalar(x, coeffs::AbstractVector)
Evaluate polynomial with coefficients `coeffs` at a single point `x`.
AD-friendly scalar version - no array allocation.
"""
@inline function evaluate_polynomial_horner_scalar(
        x::T,
        coeffs::AbstractVector{S},
    ) where {T, S}
    result = zero(promote_type(T, S))
    @inbounds @simd for coeff in reverse(coeffs)
        result = muladd(result, x, coeff)
    end
    return result
end

"""
    derivative_coeffs(coeffs)

Compute the coefficients of the derivative of a polynomial given its coefficients.
"""
function derivative_coeffs(coeffs)
    if length(coeffs) <= 1
        return [zero(eltype(coeffs))]
    end
    return [i * coeffs[i + 1] for i in 1:(length(coeffs) - 1)]
end

"""
    evaluate_derivative_horner(x, coeffs)

Evaluate the derivative of a polynomial at point x using Horner's method.
"""
function evaluate_derivative_horner(x, coeffs)
    n = length(coeffs) - 1
    if n == 0
        return 0.0
    end
    derivative_coeffs = [i * coeffs[i + 1] for i in 1:n]
    if isempty(derivative_coeffs)
        return 0.0
    end
    # Apply Horner's method
    derivative_value = derivative_coeffs[end]
    @simd for i in (n - 1):-1:1
        derivative_value = derivative_value * x + derivative_coeffs[i]
    end
    if isnan(derivative_value) || isinf(derivative_value)
        @warn "NaN or Inf in the derivative"
    end
    return derivative_value
end


"""
    numberPolynomials(n, d)

Calculate the number of polynomials of dimension n and degree d.
Uses binomial coefficient formula.
"""
function numberPolynomials(n, d)
    x, y = max(d, n), min(d, n)
    return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end

# ===== POLYNOMIAL DEGREE FUNCTIONS =====

"""
    aPCE_MultivariatePolynomialDegrees(num_dimensions, max_degree, s_marginals, s_interactions)

Generate multivariate polynomial degrees for aPCE expansion.
- `num_dimensions`: Number of input dimensions
- `max_degree`: Maximum polynomial degree
- `s_marginals`: Fraction of marginal terms to keep (0.0 to 1.0)
- `s_interactions`: Fraction of interaction terms to keep (0.0 to 1.0)
"""
function aPCE_MultivariatePolynomialDegrees(
        num_dimensions::T,
        max_degree::T,
        s_marginals::F,
        s_interactions::F,
    ) where {T <: Integer, F <: Real}
    function get_stats(r)::Array{F}
        o = Series(Mean(), Variance(), Extrema())
        n = length(r)
        summe = 0
        n_zeros = 0
        @inbounds for e in 1:n
            if Base.iszero(r[e])
                n_zeros += 1
            else
                summe += r[e]
            end
            fit!(o, r[e])
        end
        meanval = summe / n
        meanval, varval, mm = value(o)
        return [summe, n_zeros, meanval, varval, mm.min, mm.max]
    end

    function filter_by_percentage(array::AbstractArray, percentage)
        # Validate up front instead of catching the BoundsError/InexactError a
        # bad percentage would have triggered. An out-of-range (or NaN)
        # percentage keeps the whole array, matching the previous catch branch;
        # the warning fires for out-of-range but not NaN, as before.
        if !(0.0 <= percentage <= 1.0)
            isnan(percentage) || @warn "Percentage must be between 0 and 1"
            return array[:]
        end
        n = length(array)
        num_to_keep = round(Int, percentage * n)
        return @views array[1:num_to_keep]
    end

    range_ = 0:max_degree
    indices = reshape(range_, :, 1)
    for di in 1:(num_dimensions - 1)
        indices = repeat(indices, inner = (max_degree + 1, 1))
        front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di)
        indices = ApplyArray(hcat, front, indices)
        indices = @~ indices[vec(sum(indices; dims = 2)) .<= max_degree, :]
    end

    stats = reduce(hcat, map(x -> get_stats(x), eachrow(indices)))'
    d_marginal_indices = T[]
    d_interactions_indices = T[]
    @inbounds for r in axes(stats, 1)
        if stats[r, 1] <= max_degree
            if stats[r, 2] == (num_dimensions - 1)
                push!(d_marginal_indices, r)
            elseif (num_dimensions - stats[r, 2]) >= 1
                push!(d_interactions_indices, r)
            end
        end
    end

    sorting_d_marginal = stats[d_marginal_indices, 3:4]
    sorting_d_interactions = stats[d_interactions_indices, 3:4]
    all_marginals = 1:length(d_marginal_indices) |> collect
    all_interactions = 1:length(d_interactions_indices) |> collect

    if length(all_marginals) > 1
        special_sort_two_arrays!(sorting_d_marginal, all_marginals)
    end
    if length(all_interactions) > 1
        special_sort_two_arrays!(sorting_d_interactions, all_interactions)
    end

    keeper_marginals = d_marginal_indices[filter_by_percentage(all_marginals, s_marginals)]
    keeper_interactions =
        d_interactions_indices[filter_by_percentage(all_interactions, s_interactions)]
    idxkeep = vcat(keeper_marginals, keeper_interactions)
    indices = @views indices[idxkeep, :]
    indices = vcat(indices, Base.zeros(T, num_dimensions)')
    indices =
        sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
    return indices::Matrix{T}
end

# ===== BASIS FUNCTIONS =====

function solve_linear_robust(A, b; kwargs...)
    # Condition-based replacement for `try A \ b catch ... end`: select the
    # solve path explicitly instead of catching a thrown SingularException, so
    # the function stays differentiable / kernel-safe. Falls back to the
    # Levenberg-Marquardt solver in exactly the cases the direct solve fails.
    if size(A, 1) == size(A, 2)
        F = LinearAlgebra.lu(A; check = false)        # square: LU like `A \ b`
        LinearAlgebra.issuccess(F) && return F \ b
    else
        x = A \ b                                     # tall/wide: QR least squares
        all(isfinite, x) && return x
    end
    return unwrap(_solve_levenberg_marquardt_solver(A, b; kwargs...))
end


"""
    aPCE_OrthonormalBasis(Data, Degree, center_data::Val)

Compute orthonormal polynomial basis for 1D data using moment-based approach.
- `Data`: 1D array of data points
- `Degree`: Maximum polynomial degree
- `center_data`: Val{true} for centered basis, Val{false} for uncentered
"""
# function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S, center_data::Val{true}) where {T<:Real,S<:Integer}
#     if ndims(Data) > 1 && size(Data, 2) > 1
#         throw(ArgumentError("aPCE_OrthonormalBasis expects 1D data only. Use create_basis for multi-dimensional data."))
#     end

#     data_vec = vec(Data)
#     NumberOfDataPoints = length(data_vec)
#     dd = Degree

#     # Centering: scale data by mean (not subtraction)
#     data_mean = StatsBase.mean(data_vec)
#     Data_scaled = data_vec ./ data_mean

#     # Compute moments using scaled data
#     m = zeros(T, 2 * dd + 2)
#     for i in 0:(2*dd+1)
#         m[i+1] = sum(Data_scaled .^ i) / NumberOfDataPoints
#     end

#     OrthonormalBasis = zeros(T, dd + 1, dd + 1)
#     OrthogonalBasis = zeros(T, dd + 1, dd + 1)

#     for degree in 0:dd
#         Hankel = @views OrthogonalBasis[1:(degree+1), 1:(degree+1)]
#         Vc = zeros(T, degree + 1)

#         # Build Hankel matrix
#         for i in 0:(degree-1)
#             for j in 0:degree
#                 Hankel[i+1, j+1] = m[i+j+1]
#             end
#             max_val = maximum(abs.(@views Hankel[i+1, :]))
#             if max_val > eps(T) # zero(T)
#                 Hankel[i+1, :] = @views Hankel[i+1, :] / max_val
#             end
#         end

#         # Last row setup
#         for j in 0:(degree-1)
#             Hankel[degree+1, j+1] = zero(T)
#         end
#         Hankel[degree+1, degree+1] = one(T)
#         max_val = maximum(abs.(@views Hankel[degree+1, :]))
#         if max_val > eps(T)
#             Hankel[degree+1, :] = @views Hankel[degree+1, :] / max_val
#         end

#         # Right-hand side vector
#         for i in 0:(degree-1)
#             Vc[i+1] = zero(T)
#         end
#         Vc[degree+1] = one(T)

#         # Solve linear system using robust, differentiable solver
#         OrthogonalBasis[degree+1, 1:(degree+1)] .= solve_linear_robust(Hankel, Vc; λ_init=1e-3, max_iter=100)

#         # Normalization
#         P_norm = zero(T)
#         for i in 1:NumberOfDataPoints
#             Poly = zero(T)
#             for k in 0:degree
#                 Poly += @views OrthogonalBasis[degree+1, k+1] * Data_scaled[i]^k
#             end
#             P_norm += Poly^2 / NumberOfDataPoints
#         end

#         # Improved numerical stability check
#         eps_val = eps(T) * 1.0e10 # More conservative epsilon
#         if P_norm <= eps_val
#             # Fallback: use standard monomial basis for this degree
#             for i in 1:(degree+1)
#                 for j in 1:(degree+1)
#                     if i == degree + 1 && j == degree + 1
#                         OrthonormalBasis[i, j] = one(T)
#                     elseif i >= j && i <= degree
#                         OrthonormalBasis[i, j] = one(T)
#                     else
#                         OrthonormalBasis[i, j] = zero(T)
#                     end
#                 end
#             end
#         else
#             # Normal normalization with safety check
#             norm_factor = sqrt(P_norm)
#             if norm_factor <= eps_val
#                 norm_factor = eps_val
#             end
#             for k in 0:degree
#                 OrthonormalBasis[degree+1, k+1] = @views OrthogonalBasis[degree+1, k+1] / norm_factor
#             end
#         end
#     end

#     # Backward transformation to data space
#     for k in 1:size(OrthonormalBasis, 2)
#         OrthonormalBasis[:, k] ./= (data_mean^(k - 1))
#     end

#     return OrthonormalBasis
# end

# function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S, center_data::Val{false}) where {T<:Real,S<:Integer}

#     if ndims(Data) > 1 && size(Data, 2) > 1
#         # Replace this by ErrorTypes
#         throw(ArgumentError("aPCE_OrthonormalBasis expects 1D data only. Use create_basis for multi-dimensional data."))
#     end #

#     data_vec = vec(Data)
#     NumberOfDataPoints = length(data_vec)
#     dd = Degree

#     # Compute raw moments (no centering)
#     m = zeros(T, 2 * dd + 2)
#     for i in 0:(2*dd+1)
#         m[i+1] = sum(data_vec .^ i) / NumberOfDataPoints
#     end

#     OrthonormalBasis = zeros(T, dd + 1, dd + 1)
#     OrthogonalBasis = zeros(T, dd + 1, dd + 1)

#     # Same algorithm as centered version but without scaling
#     for degree in 0:dd
#         Hankel = @views OrthogonalBasis[1:(degree+1), 1:(degree+1)]
#         Vc = zeros(T, degree + 1)

#         for i in 0:(degree-1)
#             for j in 0:degree
#                 Hankel[i+1, j+1] = m[i+j+1]
#             end
#             max_val = maximum(abs.(@views Hankel[i+1, :]))
#             if max_val > zero(T)
#                 Hankel[i+1, :] = @views Hankel[i+1, :] / max_val
#             end
#         end

#         for j in 0:(degree-1)
#             Hankel[degree+1, j+1] = zero(T)
#         end
#         Hankel[degree+1, degree+1] = one(T)
#         max_val = maximum(abs.(@views Hankel[degree+1, :]))
#         if max_val > zero(T)
#             Hankel[degree+1, :] = @views Hankel[degree+1, :] / max_val
#         end

#         for i in 0:(degree-1)
#             Vc[i+1] = zero(T)
#         end
#         Vc[degree+1] = one(T)

#         OrthogonalBasis[degree+1, 1:(degree+1)] .= solve_linear_robust(Hankel, Vc)

#         # Normalization using original data
#         P_norm = zero(T)
#         for i in 1:NumberOfDataPoints
#             Poly = zero(T)
#             for k in 0:degree
#                 Poly += @views OrthogonalBasis[degree+1, k+1] * data_vec[i]^k
#             end
#             P_norm += Poly^2 / NumberOfDataPoints
#         end

#         # Improved numerical stability check
#         eps_val = eps(T) * 1.0e4 # More conservative epsilon
#         if P_norm <= eps_val
#             # Fallback: use standard monomial basis for this degree
#             for i in 1:(degree+1)
#                 for j in 1:(degree+1)
#                     if i == degree + 1 && j == degree + 1
#                         OrthonormalBasis[i, j] = one(T)
#                     elseif i >= j && i <= degree
#                         OrthonormalBasis[i, j] = one(T)
#                     else
#                         OrthonormalBasis[i, j] = zero(T)
#                     end
#                 end
#             end
#         else
#             # Normal normalization with safety check
#             norm_factor = sqrt(P_norm)
#             if norm_factor <= eps_val
#                 norm_factor = eps_val
#             end
#             for k in 0:degree
#                 OrthonormalBasis[degree+1, k+1] = @views OrthogonalBasis[degree+1, k+1] / norm_factor
#             end
#         end
#     end

#     return OrthonormalBasis
# end
# using LinearAlgebra
# using Statistics

# """
#     aPCE_OrthonormalBasis(Data, Degree, [center_data])

# Constructs orthonormal polynomial basis coefficients using the Discretized Stieltjes procedure.

# # Arguments
# - `Data::AbstractArray{T}`: 1D array of data samples.
# - `Degree::Integer`: Maximum polynomial degree.
# - `center_data::Val{Bool}`: If `Val{true}`, data is standardized internally (recommended).

# # Returns
# - Matrix `Coeffs` where `Coeffs[i, :]` are the coefficients for the polynomial of degree `i-1`.
#   Coefficients are in the monomial basis: P_k(x) = sum_j Coeffs[k+1, j+1] * x^j
# """
# function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::Integer, ::Val{CenterData}=Val(true)) where {T<:Real,CenterData}
#     if ndims(Data) > 1 && size(Data, 2) > 1
#         throw(ArgumentError("aPCE_OrthonormalBasis expects 1D data. Flatten your input."))
#     end

#     x_raw = vec(Data)
#     N = length(x_raw)
#     D = Degree

#     # Standardization
#     if CenterData
#         μ = mean(x_raw)
#         σ_val = std(x_raw; mean=μ)
#         σ = σ_val > 10 * eps(T) ? σ_val : one(T)
#     else
#         μ = zero(T)
#         σ = one(T)
#     end

#     x = (x_raw .- μ) ./ σ

#     # Storage for polynomial evaluations at all data points
#     # P_evals[k+1, i] = P_k(x[i])
#     P_evals = zeros(T, D + 1, N)
#     P_evals[1, :] .= one(T)  # P_0 = 1

#     # Storage for monic polynomial coefficients in transformed variable
#     # MonicCoeffs[k+1, j+1] = coefficient of x^j in monic P_k
#     MonicCoeffs = zeros(T, D + 1, D + 1)
#     MonicCoeffs[1, 1] = one(T)  # P_0 = 1

#     # Recurrence coefficients
#     α = zeros(T, D)
#     β = zeros(T, D)
#     norms_sq = zeros(T, D + 1)  # <P_k, P_k>

#     norms_sq[1] = T(N)  # <P_0, P_0> = N

#     # Stieltjes recurrence: P_{k+1}(x) = (x - α_k) P_k(x) - β_k P_{k-1}(x)
#     for k in 0:(D-1)
#         p_k = @view P_evals[k+1, :]

#         # α_k = <x P_k, P_k> / <P_k, P_k>
#         pk_sq_norm = dot(p_k, p_k)
#         norms_sq[k+1] = pk_sq_norm

#         α[k+1] = dot(x .* p_k, p_k) / pk_sq_norm

#         # β_k = <P_k, P_k> / <P_{k-1}, P_{k-1}>
#         if k == 0
#             β[k+1] = pk_sq_norm  # β_0 not used in recurrence but store for reference
#         else
#             β[k+1] = pk_sq_norm / norms_sq[k]
#         end

#         # Compute P_{k+1} evaluations
#         p_kp1 = @view P_evals[k+2, :]
#         @. p_kp1 = (x - α[k+1]) * p_k
#         if k > 0
#             p_km1 = @view P_evals[k, :]
#             @. p_kp1 -= β[k+1] * p_km1
#         end

#         # Update coefficient matrix: P_{k+1} = x*P_k - α_k*P_k - β_k*P_{k-1}
#         # x * P_k: shift coefficients right
#         for j in 0:k
#             MonicCoeffs[k+2, j+2] += MonicCoeffs[k+1, j+1]
#         end
#         # -α_k * P_k
#         for j in 0:k
#             MonicCoeffs[k+2, j+1] -= α[k+1] * MonicCoeffs[k+1, j+1]
#         end
#         # -β_k * P_{k-1}
#         if k > 0
#             for j in 0:(k-1)
#                 MonicCoeffs[k+2, j+1] -= β[k+1] * MonicCoeffs[k, j+1]
#             end
#         end
#     end

#     # Final norm
#     norms_sq[D+1] = dot(P_evals[D+1, :], P_evals[D+1, :])

#     # Transform coefficients back to original variable and normalize
#     # If P_k(x') with x' = (x - μ)/σ, then in terms of x:
#     # (x')^p = ((x - μ)/σ)^p = σ^{-p} * sum_{j=0}^p binom(p,j) * x^j * (-μ)^{p-j}

#     OrthonormalBasis = zeros(T, D + 1, D + 1)

#     for k in 0:D
#         # Normalization factor: 1/sqrt(<P_k, P_k>/N)
#         norm_factor = sqrt(norms_sq[k+1] / N)
#         norm_factor = norm_factor < eps(T) ? one(T) : norm_factor

#         # For each power p in the monic polynomial
#         for p in 0:k
#             c = MonicCoeffs[k+1, p+1] / norm_factor

#             # Expand (x')^p = ((x - μ)/σ)^p using binomial theorem
#             inv_σ_p = one(T) / (σ^p)
#             for j in 0:p
#                 binom_coeff = binomial(p, j)
#                 OrthonormalBasis[k+1, j+1] += c * inv_σ_p * binom_coeff * ((-μ)^(p - j))
#             end
#         end
#     end

#     return OrthonormalBasis
# end


### VERSION 1

# using LinearAlgebra
# using Statistics

# """
#     aPCE_OrthonormalBasis(Data, Degree, center_data)

# Constructs orthonormal polynomial basis exactly as in Oladyshkin & Nowak (2012).
# Uses Hankel matrix solve (Eq. 14) and Hankel inner product normalization (Eq. 22-23).
# """
# function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::Integer, ::Val{true}) where {T<:Real}
#     x_raw = vec(Data)

#     # Standardize (paper Eq. 15)
#     μ = mean(x_raw)
#     σ_val = std(x_raw; mean=μ)
#     σ = σ_val > eps(T) ? σ_val : one(T)
#     x = (x_raw .- μ) ./ σ

#     # Build basis in transformed space
#     OrthonormalCoeffs = _build_orthonormal_basis(x, Degree)

#     # Back-transform via binomial expansion
#     return _backtransform(OrthonormalCoeffs, μ, σ, Degree)
# end

# function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::Integer, ::Val{false}) where {T<:Real}
#     x = vec(Data)
#     return _build_orthonormal_basis(x, Degree)
# end

# function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::Integer) where {T<:Real}
#     return aPCE_OrthonormalBasis(Data, Degree, Val(true))
# end

# """
# Build orthonormal basis using Hankel method (paper Eq. 14, 22-23).
# """
# function _build_orthonormal_basis(x::AbstractVector{T}, D::Integer) where {T<:Real}
#     # Compute moments: m[k+1] = E[x^k]
#     m = zeros(T, 2D + 1)
#     for k in 0:2D
#         m[k+1] = mean(x .^ k)
#     end

#     # Monic orthogonal polynomials via Hankel solve (Eq. 14)
#     MonicCoeffs = zeros(T, D + 1, D + 1)
#     MonicCoeffs[1, 1] = one(T)  # P_0 = 1

#     for k in 1:D
#         # Build moment matrix
#         M = zeros(T, k + 1, k + 1)
#         rhs = zeros(T, k + 1)

#         # Orthogonality conditions: Σᵢ pᵢ m_{i+j} = 0 for j = 0..k-1
#         for row in 0:(k-1)
#             for col in 0:k
#                 M[row+1, col+1] = m[row+col+1]
#             end
#         end

#         # Monic constraint: p_k = 1
#         M[k+1, k+1] = one(T)
#         rhs[k+1] = one(T)

#         MonicCoeffs[k+1, 1:(k+1)] = M \ rhs
#     end

#     # Normalize via Hankel inner product (Eq. 22-23)
#     # ‖P_k‖² = Σᵢ Σⱼ pᵢ pⱼ m_{i+j}
#     OrthonormalCoeffs = zeros(T, D + 1, D + 1)

#     for k in 0:D
#         norm_sq = zero(T)
#         for i in 0:k
#             for j in 0:k
#                 norm_sq += MonicCoeffs[k+1, i+1] * MonicCoeffs[k+1, j+1] * m[i+j+1]
#             end
#         end

#         norm_factor = sqrt(max(norm_sq, eps(T)))

#         for j in 0:k
#             OrthonormalCoeffs[k+1, j+1] = MonicCoeffs[k+1, j+1] / norm_factor
#         end
#     end

#     return OrthonormalCoeffs
# end

# """
# Back-transform from standardized variable x' = (x-μ)/σ to original x.
# """
# function _backtransform(Coeffs::AbstractMatrix{T}, μ::T, σ::T, D::Integer) where {T<:Real}
#     Result = zeros(T, D + 1, D + 1)

#     for k in 0:D
#         for j in 0:k
#             c = Coeffs[k+1, j+1]
#             inv_σ_j = one(T) / (σ^j)
#             for l in 0:j
#                 Result[k+1, l+1] += c * inv_σ_j * binomial(j, l) * ((-μ)^(j - l))
#             end
#         end
#     end

#     return Result
# end

#### Version2
"""
    aPCE_OrthonormalBasis(Data, Degree, center_data)

Constructs orthonormal polynomial basis using the Stieltjes procedure.
Mathematically equivalent to Oladyshkin & Nowak (2012) Hankel method.
"""
function aPCE_OrthonormalBasis(
        Data::AbstractArray{T},
        Degree::Integer,
        ::Val{true},
    ) where {T <: Real}
    x_raw = vec(Data)
    N = length(x_raw)
    D = Degree

    # Standardize: zero mean, unit variance (paper Eq. 15)
    μ = StatsBase.mean(x_raw)
    σ_val = StatsBase.std(x_raw; mean = μ)
    σ = σ_val > eps(T) ? σ_val : one(T)
    x = (x_raw .- μ) ./ σ

    # Build monic basis in transformed space
    MonicCoeffs = _stieltjes_core(x, D)

    # Normalize to orthonormal form via the empirical sample norm
    OrthonormalCoeffs = _normalize_hankel(MonicCoeffs, x, D)

    # Back-transform to original variable via binomial expansion
    FinalBasis = zeros(T, D + 1, D + 1)

    for k in 0:D
        for j in 0:k
            c = OrthonormalCoeffs[k + 1, j + 1]
            inv_σ_j = one(T) / (σ^j)
            for l in 0:j
                FinalBasis[k + 1, l + 1] += c * inv_σ_j * binomial(j, l) * ((-μ)^(j - l))
            end
        end
    end

    return FinalBasis
end

function aPCE_OrthonormalBasis(
        Data::AbstractArray{T},
        Degree::Integer,
        ::Val{false},
    ) where {T <: Real}
    x = vec(Data)
    D = Degree

    # No centering - work directly with raw data
    MonicCoeffs = _stieltjes_core(x, D)

    # Normalize to orthonormal form via the empirical sample norm
    return _normalize_hankel(MonicCoeffs, x, D)
end

# Default: centered
function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::Integer) where {T <: Real}
    return aPCE_OrthonormalBasis(Data, Degree, Val(true))
end

"""
Core Stieltjes three-term recurrence. Returns the monic coefficients of the
orthogonal polynomials P_0..P_D w.r.t. the empirical measure of `x`.
"""
function _stieltjes_core(x::AbstractVector{T}, D::Integer) where {T <: Real}
    N = length(x)

    # Polynomial evaluations
    P_prev = zeros(T, N)
    P_curr = ones(T, N)
    P_next = zeros(T, N)

    # Monic polynomial coefficients
    MonicCoeffs = zeros(T, D + 1, D + 1)
    MonicCoeffs[1, 1] = one(T)

    # Track inner products directly
    inner_prev = one(T)  # <P_{k-1}, P_{k-1}>, initialized for k=0

    for k in 0:(D - 1)
        # <P_k, P_k>
        inner_curr = sum(abs2, P_curr) / N

        # α_k = <x P_k, P_k> / <P_k, P_k>
        α_k = sum(i -> x[i] * P_curr[i]^2, eachindex(x)) / (N * inner_curr)

        # β_k = <P_k, P_k> / <P_{k-1}, P_{k-1}>
        β_k = inner_curr / inner_prev

        # P_{k+1} = (x - α_k) P_k - β_k P_{k-1}
        if k > 0
            @. P_next = (x - α_k) * P_curr - β_k * P_prev
        else
            @. P_next = (x - α_k) * P_curr
        end

        # Coefficient recurrence: P_{k+1} = x·P_k - α_k·P_k - β_k·P_{k-1}

        # x · P_k (shift right)
        for j in 0:k
            MonicCoeffs[k + 2, j + 2] += MonicCoeffs[k + 1, j + 1]
        end

        # -α_k · P_k
        for j in 0:k
            MonicCoeffs[k + 2, j + 1] -= α_k * MonicCoeffs[k + 1, j + 1]
        end

        # -β_k · P_{k-1}
        if k > 0
            for j in 0:(k - 1)
                MonicCoeffs[k + 2, j + 1] -= β_k * MonicCoeffs[k, j + 1]
            end
        end

        # Advance: three-way rotation keeps all three buffers distinct — the
        # in-place `@. P_next = ...` above reads P_prev/P_curr, so any aliasing
        # among them corrupts the recurrence from degree 3 onward.
        P_prev, P_curr, P_next = P_curr, P_next, P_prev
        inner_prev = inner_curr
    end

    return MonicCoeffs
end

"""
Normalize the monic orthogonal polynomials to orthonormal form using the
*empirical* L2 norm against the sample measure of `data`:

    ‖P_k‖² = mean_i P_k(x_i)²

This replaces the moment-sum Hankel form ‖P_k‖² = Σᵢ Σⱼ pᵢ pⱼ m_{i+j}, which is
mathematically equivalent but numerically catastrophic: the moment matrix H is
exponentially ill-conditioned in the degree (Gautschi), so the double sum
cancels to garbage beyond degree ~14 and yields absurd normalization factors.
Evaluating the monic polynomial directly on the data is O(n·d) per degree and is
exact w.r.t. the empirical measure the basis must be orthonormal against.
"""
function _normalize_hankel(
        MonicCoeffs::AbstractMatrix{T},
        data::AbstractVector{T},
        D::Integer,
    ) where {T <: Real}
    OrthonormalCoeffs = zeros(T, D + 1, D + 1)
    N = length(data)

    for k in 0:D
        p = @view MonicCoeffs[k + 1, 1:(k + 1)]

        # ‖P_k‖² = mean_i P_k(x_i)²  (empirical sample norm)
        norm_sq = zero(T)
        for x_i in data
            v = evalpoly_two(x_i, p)
            norm_sq += v * v
        end
        norm_sq /= N

        norm_factor = sqrt(max(norm_sq, eps(T)))

        for j in 0:k
            OrthonormalCoeffs[k + 1, j + 1] = MonicCoeffs[k + 1, j + 1] / norm_factor
        end
    end

    return OrthonormalCoeffs
end

function _solve_levenberg_marquardt_solver(
        Psi::AbstractMatrix{T},
        y::AbstractVector{T};
        λ_init = 1.0e-3,
        max_iter = 100,
    )::Result{Vector{T}, String} where {T <: Real}
    # Levenberg-Marquardt with adaptive regularization
    λ = 0.0
    x = pinv(Psi) * y # Initial guess using pinv
    λ = T(λ_init)

    for _ in 1:max_iter
        residual = Psi * x - y
        J = Psi  # Jacobian is just Psi for linear case

        # Gauss-Newton + damping
        JtJ = J' * J
        Jtr = J' * residual

        # Try step with current λ
        δx = -(JtJ + λ * I) \ Jtr
        x_new = x + δx

        new_residual = Psi * x_new - y

        # Accept/reject step and adjust λ
        if norm(new_residual) < norm(residual)
            x = x_new
            λ *= as(T, 0.3)  # Decrease damping
        else
            λ *= as(T, 2.0)  # Increase damping
        end

        if norm(δx) < as(T, 1.0e-10)
            break
        end
    end

    return Ok(x)
end

# Public solve_levenberg_marquardt function using direct solver (not implicit diff)
function solve_levenberg_marquardt(
        Psi,
        y;
        λ_init = 1.0e-3,
        max_iter = 50,
    )::Result{Vector{eltype(Psi)}, String}
    return _solve_levenberg_marquardt_solver(Psi, y; λ_init = λ_init, max_iter = max_iter)
end

function robust_iterative_refinement(A, b; maxiter = 5, tol = 1.0e-10)
    # Initial solution using pseudoinverse
    x = pinv(A) * b

    # Sanitize initial solution
    x = map(xi -> isfinite(xi) ? xi : 0.0, x)

    converged = false

    for iter in 1:maxiter
        r = b - A * x

        # Check convergence
        if norm(r) < tol
            converged = true
            break
        end

        # Compute correction using pseudoinverse for robustness
        dx = pinv(A) * r
        x_new = x + dx

        # Check if new solution is valid
        if all(isfinite, x_new)
            x = x_new
        else
            # Non-finite correction - stop refinement
            break
        end
    end
    if !converged
        @warn "Iterative refinement did not converge"
    end

    return x
end


"""
    aPCE_FullBasis(Data, Degree)

Generate a full (monomial) basis up to specified degree.
Returns an upper triangular matrix with 1s.
"""
function aPCE_FullBasis(Data, Degree)
    T = eltype(Data)
    FullBasis = Matrix{T}(undef, Degree + 1, Degree + 1)

    for i in 1:(Degree + 1)
        for j in 1:(Degree + 1)
            if i >= j
                FullBasis[i, j] = one(T)
            else
                FullBasis[i, j] = zero(T)
            end
        end
    end
    return FullBasis
end

"""
    _apce_closed_form_basis_moment!(basis, col, degree)

Compute polynomial basis using closed-form solutions for degrees 0-4.
This is more efficient than the general numerical method.
"""
function _apce_closed_form_basis_moment!(basis, col, degree)
    n = length(col)
    m = zeros(eltype(col), 2 * degree + 2)
    for i in 0:(2 * degree + 1)
        m[i + 1] = sum(col .^ i) / n
    end

    if degree == 0
        basis[1, 1] = 1
    elseif degree == 1
        basis[2, 1] = 0
        basis[2, 2] = 1
        basis[1, 1] = 1
    elseif degree == 2
        basis[3, 1] = -1
        basis[3, 2] = -m[4]
        basis[3, 3] = 1
        basis[2, 1] = 0
        basis[2, 2] = 1
        basis[1, 1] = 1
    elseif degree == 3
        denom = 1 - m[4] + m[4]^2
        basis[4, 1] = -1 / denom * (-m[4]^2 + m[4]^3 - m[4] * m[5] + m[6])
        basis[4, 2] = (-m[4] * m[6] + m[4]^2 - m[5] + m[4] * m[5]) / denom
        basis[4, 3] = -(m[4] * m[5] - m[6] + m[4]) / denom
        basis[4, 4] = 1
        basis[3, 1] = -1
        basis[3, 2] = -m[4]
        basis[3, 3] = 1
        basis[2, 1] = 0
        basis[2, 2] = 1
        basis[1, 1] = 1
    elseif degree == 4
        denom = -m[4]^3 + m[4] * m[7] - 2 * m[6] * m[5] + m[5]^2
        basis[5, 1] =
            -(
            -m[4] * m[6] * m[5] - m[4]^2 * m[8] + m[4] * m[6]^2 +
                2 * m[4] * m[5] * m[7] - 2 * m[5]^2 * m[6] + m[5]^3
        ) / denom
        basis[5, 2] =
            -(
            m[6]^3 - m[5]^2 * m[4]^2 + m[4]^2 * m[5] * m[6] - m[5] * m[6] * m[7] -
                m[4] * m[6] * m[7] - m[4] * m[6] * m[8] + m[4] * m[5] * m[8] +
                m[4]^3 * m[6] - m[4]^3 * m[7] + m[4] * m[7]^2
        ) / (m[4] * denom)
        basis[5, 3] =
            -(
            m[5]^2 * m[4]^2 - m[4] * m[5] * m[8] - m[5] * m[6]^2 + m[5]^2 * m[7] -
                m[4]^3 * m[6] + m[4] * m[6] * m[7]
        ) / (m[4] * denom)
        basis[5, 4] =
            (m[4]^2 * m[5] - m[6] * m[5] - m[4] * m[8] + m[6]^2 + m[5] * m[7]) / denom
        basis[5, 5] = 1
        basis[4, 1] = -1 / (1 - m[4] + m[4]^2) * (-m[4]^2 + m[4]^3 - m[4] * m[5] + m[6])
        basis[4, 2] = (-m[4] * m[6] + m[4]^2 - m[5] + m[4] * m[5]) / (1 - m[4] + m[4]^2)
        basis[4, 3] = -(m[4] * m[5] - m[6] + m[4]) / (1 - m[4] + m[4]^2)
        basis[4, 4] = 1
        basis[3, 1] = -1
        basis[3, 2] = -m[4]
        basis[3, 3] = 1
        basis[2, 1] = 0
        basis[2, 2] = 1
        basis[1, 1] = 1
    else
        error("Closed-form aPC basis only implemented for degree ≤ 4")
    end
    return basis
end

"""
    create_basis(x, degree, is_orthonormal::Bool; center_data = true)

Convenience method that converts Bool to Val for dispatch.
"""
function create_basis(x, degree, is_orthonormal::Bool; center_data::Bool = true)
    if is_orthonormal
        return create_basis(x, degree, Val(true); center_data = center_data)
    else
        return create_basis(x, degree, Val(false); center_data = center_data)
    end
end

# ============================================================================
# CenteredBasis: z-space polynomial basis + per-dimension (μ, σ) statistics
#
# Rationale: the closed-form aPCE coefficients for degrees 0-4 use raw moments
# m[k] = E[x^k]. On raw data with large σ (e.g., ackley_5d has σ ≈ 18), the
# higher moments m[6], m[7], m[8] grow as σ^k and overwhelm numerical
# precision, producing ill-conditioned polynomial coefficients. The Stieltjes
# path (degree > 4) works internally in z-space but then back-transforms to
# x-space via binomial expansion (σ^{-j} · (-μ)^{j-l}), which is numerically
# delicate and produces gradients that collapse Mooncake AD stability.
#
# CenteredBasis defers standardization to evaluation time: the polynomial
# coefficients are stored in z-space (well-conditioned), and (μ, σ) are
# applied as elementary arithmetic at the Psi-matrix stage, where ChainRules /
# Mooncake handle the gradient cleanly.
# ============================================================================

"""
    CenteredBasis{T, A}

Opt-in wrapper carrying a z-space polynomial basis and the per-dimension
`(μ, σ)` statistics used to standardize raw inputs before evaluation.

Fields:
- `basis::A`: orthonormal polynomial coefficients in z-space, shape
  `(degree + 1, degree + 1, input_dimensions)`.
- `μ::Vector{T}`: per-dimension means, length `input_dimensions`.
- `σ::Vector{T}`: per-dimension standard deviations (never zero;
  degenerate dimensions fall back to `1`), length `input_dimensions`.

Typical construction via [`create_centered_basis`](@ref). Evaluation via
`aPCE_PsiPolynomialMatrix_zygote(x, degrees, cb)` which standardizes `x`
per-dimension and delegates to the `AbstractArray` method on `cb.basis`.
"""
struct CenteredBasis{T <: Real, A <: AbstractArray{T}}
    basis::A
    μ::Vector{T}
    σ::Vector{T}
end

Base.eltype(::Type{<:CenteredBasis{T}}) where {T} = T
Base.eltype(cb::CenteredBasis) = eltype(typeof(cb))
Base.size(cb::CenteredBasis, args...) = size(cb.basis, args...)
Base.ndims(cb::CenteredBasis) = ndims(cb.basis)

"""
    create_centered_basis(x, degree)

Build an orthonormal polynomial basis in z-space and return it together with
the per-dimension standardization statistics as a [`CenteredBasis`](@ref).

This is an opt-in alternative to `create_basis(x, degree; center_data=true)`
that avoids the x-space binomial back-transform (which is numerically
unstable and gradient-unfriendly for large σ). Use it when you need the
closed-form / Stieltjes basis to be well-conditioned on raw data without
requiring the caller to pre-standardize their inputs.

The caller does not have to standardize `x`; evaluation functions dispatch on
`CenteredBasis` and apply `(x .- μ') ./ σ'` before evaluating polynomials.
"""
function create_centered_basis(
        x::AbstractArray{T},
        degree::Integer,
    ) where {T <: Real}
    if ndims(x) == 1
        x = reshape(x, :, 1)
    end

    input_dimensions = size(x, 2)
    μ = zeros(T, input_dimensions)
    σ = zeros(T, input_dimensions)
    basis = zeros(T, degree + 1, degree + 1, input_dimensions)

    for i in 1:input_dimensions
        col = x[:, i]
        μ_i = StatsBase.mean(col)
        σ_val = StatsBase.std(col; mean = μ_i)
        σ_i = σ_val > eps(T) ? σ_val : one(T)
        μ[i] = μ_i
        σ[i] = σ_i
        z = (col .- μ_i) ./ σ_i

        # Stieltjes in z-space for ALL degrees: reuse the core recurrence +
        # empirical normalization, but STOP before the binomial back-transform.
        #
        # The closed-form path (`_apce_closed_form_basis_moment!`) is NOT used
        # here: it returns *monic* orthogonal polynomials (leading coeff 1, not
        # unit norm), so for degree ≥ 2 the resulting basis is orthogonal but
        # not orthonormal and Ψ'Ψ/n ≠ I. Routing every degree through the
        # normalized Stieltjes recurrence keeps the d ≤ 4 and d ≥ 5 bases
        # consistent and genuinely orthonormal. The recurrence is cheap at
        # d ≤ 4 and, with the empirical-norm fix, well-conditioned through the
        # degrees this package is used at (≤ ~12); beyond that the monomial
        # coefficient representation itself limits accuracy (see below).
        #
        # NOTE on very high degree: the basis is stored as monomial coefficients
        # and evaluated with `evalpoly`, so predicting at new points needs those
        # coefficients. For a strongly skewed measure the monomial form of a
        # degree ≳ 16 orthonormal polynomial has large alternating coefficients
        # and loses precision on evaluation — a limit of the representation, not
        # of the normalization. (A three-term-recurrence storage/eval scheme
        # would be needed to push materially past that.)
        MonicCoeffs = _stieltjes_core(z, degree)
        OrthonormalCoeffs = _normalize_hankel(MonicCoeffs, z, degree)
        basis[:, :, i] .= OrthonormalCoeffs
    end

    return CenteredBasis{T, typeof(basis)}(basis, μ, σ)
end

# ============================================================================
# RecurrenceCenteredBasis: three-term-recurrence (Jacobi-matrix) basis
#
# An opt-in alternative to `CenteredBasis` that stores the recurrence
# coefficients (αₖ, βₖ) of the orthonormal polynomials instead of their
# monomial coefficients, and evaluates them via the stable forward three-term
# recurrence rather than `evalpoly`. This removes the monomial-representation
# ceiling that caps `CenteredBasis` orthonormality at degree ≈ 12 for strongly
# skewed measures: every orthonormal value stays O(1), so there is no
# catastrophic cancellation and the basis stays orthonormal to much higher
# degree. Cost is the same O(n·d) as `evalpoly`; the trade-off is that new
# evaluation points must be run through the recurrence (no closed monomial
# form). Cf. RecurrenceRelationships.jl's `forwardrecurrence`.
# ============================================================================

"""
    RecurrenceCenteredBasis{T}

Opt-in basis carrying the orthonormal three-term-recurrence coefficients in
z-space and the per-dimension `(μ, σ)` standardization statistics.

Fields:
- `α::Matrix{T}`: recurrence diagonal, shape `(degree, input_dimensions)`
  (`α₀ … α_{degree-1}` per dimension).
- `β::Matrix{T}`: recurrence squared off-diagonal, shape
  `(degree + 1, input_dimensions)` (`β₀ … β_degree` per dimension, `β₀ = 1`).
- `degree::Int`: polynomial degree.
- `μ::Vector{T}` / `σ::Vector{T}`: per-dimension standardization statistics.

Construct via [`create_recurrence_basis`](@ref); evaluate via
`aPCE_PsiPolynomialMatrix_zygote(x, degrees, rb)`.
"""
struct RecurrenceCenteredBasis{T <: Real}
    α::Matrix{T}
    β::Matrix{T}
    degree::Int
    μ::Vector{T}
    σ::Vector{T}
end

Base.eltype(::Type{<:RecurrenceCenteredBasis{T}}) where {T} = T
Base.eltype(rb::RecurrenceCenteredBasis) = eltype(typeof(rb))
# Reported shape mirrors the equivalent monomial-coefficient tensor
# `(degree + 1, degree + 1, input_dimensions)` so `show(::aPCE)` and any
# size-based introspection work uniformly across basis backends.
Base.size(rb::RecurrenceCenteredBasis) = (rb.degree + 1, rb.degree + 1, size(rb.α, 2))
Base.size(rb::RecurrenceCenteredBasis, d::Integer) = size(rb)[d]

"""
    _stieltjes_recurrence(x, D) -> (α, β)

Run the Stieltjes procedure and return the orthonormal three-term-recurrence
coefficients w.r.t. the empirical measure of `x`:

    √β_{k+1} p_{k+1}(x) = (x - α_k) p_k(x) - √β_k p_{k-1}(x),  p_0 = 1/√β_0

`α` has length `D` (α₀ … α_{D-1}); `β` has length `D + 1` (β₀ … β_D, β₀ = 1).
These are exactly the coefficients the monic recurrence in `_stieltjes_core`
already forms internally — here we keep them instead of expanding to monomials.
"""
function _stieltjes_recurrence(x::AbstractVector{T}, D::Integer) where {T <: Real}
    N = length(x)
    α = zeros(T, max(D, 0))
    β = zeros(T, D + 1)
    β[1] = one(T)                       # β₀ = ⟨P₀, P₀⟩ = 1 (P₀ ≡ 1)

    P_prev = zeros(T, N)
    P_curr = ones(T, N)
    P_next = zeros(T, N)
    inner_prev = one(T)                 # ⟨P_{k-1}, P_{k-1}⟩, seeded for k = 0

    for k in 0:(D - 1)
        inner_curr = sum(abs2, P_curr) / N          # ⟨P_k, P_k⟩
        α[k + 1] = sum(i -> x[i] * P_curr[i]^2, eachindex(x)) / (N * inner_curr)
        
        if k > 0
            β[k + 1] = inner_curr / inner_prev                      # β_k
            @. P_next = (x - α[k + 1]) * P_curr - β[k + 1] * P_prev
        else
            @. P_next = (x - α[k + 1]) * P_curr
        end

        P_prev, P_curr, P_next = P_curr, P_next, P_prev
        inner_prev = inner_curr
    end

    if D >= 1
        inner_D = sum(abs2, P_curr) / N             # ⟨P_D, P_D⟩
        β[D + 1] = inner_D / inner_prev                             # β_D
    end

    return α, β
end

"""
    create_recurrence_basis(x, degree)

Build an orthonormal polynomial basis in z-space stored as its three-term
recurrence coefficients, returned as a [`RecurrenceCenteredBasis`](@ref).

This is the opt-in, high-degree-stable counterpart to
[`create_centered_basis`](@ref): identical z-space standardization contract,
but evaluation goes through the forward recurrence (see
`aPCE_PsiPolynomialMatrix_zygote(x, degrees, rb)`), so orthonormality does not
degrade at high degree the way the monomial-coefficient basis does.
"""
function create_recurrence_basis(
        x::AbstractArray{T},
        degree::Integer,
    ) where {T <: Real}
    if ndims(x) == 1
        x = reshape(x, :, 1)
    end

    input_dimensions = size(x, 2)
    μ = zeros(T, input_dimensions)
    σ = zeros(T, input_dimensions)
    α = zeros(T, max(degree, 0), input_dimensions)
    β = zeros(T, degree + 1, input_dimensions)

    for i in 1:input_dimensions
        col = x[:, i]
        μ_i = StatsBase.mean(col)
        σ_val = StatsBase.std(col; mean = μ_i)
        σ_i = σ_val > eps(T) ? σ_val : one(T)
        μ[i] = μ_i
        σ[i] = σ_i
        z = (col .- μ_i) ./ σ_i

        α_i, β_i = _stieltjes_recurrence(z, degree)
        α[:, i] .= α_i
        β[:, i] .= β_i
    end

    return RecurrenceCenteredBasis{T}(α, β, Int(degree), μ, σ)
end


"""
    create_basis(x, degree; center_data = true)

Default basis creation function. Creates orthonormal basis with optional centering.
"""
function create_basis(x, degree; center_data::Bool = true)
    return create_basis(x, degree, Val(true); center_data = center_data)
end

"""
    create_basis(x, degree, is_orthonormal::Val{true}; center_data = true)

Create orthonormal basis for multi-dimensional data.
Uses closed-form solutions for degrees 0-4, numerical method for higher degrees.
"""
function create_basis(
        x::AbstractArray{T},
        degree::Integer,
        ::Val{true};
        center_data::Bool = true,
    ) where {T <: Real}
    if ndims(x) == 1
        x = reshape(x, :, 1)
    end

    input_dimensions = size(x, 2)
    OrthonormalBasis = zeros(T, degree + 1, degree + 1, input_dimensions)

    if degree in 0:4
        for i in 1:input_dimensions
            col = x[:, i]
            basis_slice = zeros(T, degree + 1, degree + 1)
            _apce_closed_form_basis_moment!(basis_slice, col, degree)
            OrthonormalBasis[:, :, i] .= basis_slice
        end
    else
        for i in 1:input_dimensions
            col_values = x[:, i]
            result = aPCE_OrthonormalBasis(col_values, degree, Val(center_data))
            OrthonormalBasis[:, :, i] .= result
        end
    end

    return OrthonormalBasis
end

"""
    create_basis(x, degree, ::Val{false}; center_data = false)

Create full (monomial) basis for multi-dimensional data.
Note: center_data parameter is ignored for monomial basis.
"""
function create_basis(
        x::AbstractArray{T},
        degree::Integer,
        ::Val{false};
        center_data::Bool = false,
    ) where {T <: Real}
    if ndims(x) == 1
        x = reshape(x, :, 1)
    end

    input_dimensions = size(x, 2)
    FullBasis = zeros(T, degree + 1, degree + 1, input_dimensions)

    for i in 1:input_dimensions
        col_values = view(x, :, i)
        FullBasis[:, :, i] .= aPCE_FullBasis(col_values, degree)
    end
    return FullBasis
end


# ===== PSI MATRIX FUNCTIONS =====

"""
    compute_Psi_element(i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions)

Compute a single element of the Psi matrix (polynomial evaluation).
"""
function compute_Psi_element(
        i,
        j,
        TrainingInput,
        MultivariatePolynomialDegrees,
        OrthonormalBasis,
        InputDimensions,
    )
    product = one(eltype(TrainingInput))
    @inbounds for ii in 1:InputDimensions
        degree = MultivariatePolynomialDegrees[i, ii] + 1
        coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
        x = TrainingInput[j, ii]
        product *= evalpoly_two(x, coeffs)
    end
    return product
end

"""
    aPCE_PsiPolynomialMatrix_zygote(TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)

Generic fallback version for non-typed inputs.
"""
function aPCE_PsiPolynomialMatrix_zygote(
        TrainingInput,
        MultivariatePolynomialDegrees,
        OrthonormalBasis,
    )
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)

    Psi = [
        compute_Psi_element(
                i,
                j,
                TrainingInput,
                MultivariatePolynomialDegrees,
                OrthonormalBasis,
                InputDimensions,
            )
            for i in 1:NumberOfTerms, j in 1:NCpoints
    ]
    return reshape(Psi, NumberOfTerms, NCpoints)
end

"""
    aPCE_PsiPolynomialMatrix_zygote(TrainingInput, MultivariatePolynomialDegrees, cb::CenteredBasis)

CenteredBasis dispatch: standardize `TrainingInput` per-dimension using the
`(μ, σ)` captured at basis construction, then delegate to the `AbstractArray`
method on `cb.basis`. Standardization is elementary broadcasting, so both
ChainRules and Mooncake handle its gradient without the x-space binomial
back-transform that destabilizes closed-form / Stieltjes coefficients on
raw data with large σ.
"""
function aPCE_PsiPolynomialMatrix_zygote(
        TrainingInput,
        MultivariatePolynomialDegrees,
        cb::CenteredBasis,
    )
    μ_row = reshape(cb.μ, 1, :)
    σ_row = reshape(cb.σ, 1, :)
    z = (TrainingInput .- μ_row) ./ σ_row
    return aPCE_PsiPolynomialMatrix_zygote(z, MultivariatePolynomialDegrees, cb.basis)
end

"""
    _orthonormal_recurrence_values(z, α, β, D) -> Matrix

Evaluate the orthonormal polynomials `p_0 … p_D` at every point of `z` via the
forward three-term recurrence, returning a `(length(z), D + 1)` matrix whose
column `k + 1` is `p_k` on the data. Non-mutating (each column is a fresh
broadcast) so it stays ForwardDiff / Mooncake differentiable w.r.t. `z`.
"""
function _orthonormal_recurrence_values(
        z::AbstractVector{Tz},
        α::AbstractVector,
        β::AbstractVector,
        D::Integer,
    ) where {Tz}
    N = length(z)
    out = Matrix{Tz}(undef, N, D + 1)
    @inbounds for i in 1:N
        zi = z[i]
        p0 = one(Tz) / sqrt(Tz(β[1]))
        out[i, 1] = p0
        if D > 0
            p1 = (zi - α[1]) * p0 / sqrt(Tz(β[2]))
            out[i, 2] = p1
            for k in 1:(D - 1)
                out[i, k + 2] = ((zi - α[k + 1]) * out[i, k + 1] - sqrt(Tz(β[k + 1])) * out[i, k]) / sqrt(Tz(β[k + 2]))
            end
        end
    end
    return out
end

"""
    _orthonormal_recurrence_derivative_values(z, α, β, degree, max_order) -> Array{Tz, 3}

Evaluate the orthonormal polynomials `p_0 … p_degree` and their `z`-derivatives
up to order `max_order` at every point of `z` via the differentiated forward
three-term recurrence. Returns an `(length(z), degree + 1, max_order + 1)`
array whose entry `[i, k + 1, m + 1]` is `∂ᵐ p_k / ∂zᵐ` at `z[i]`; the `m = 0`
slice coincides exactly with
[`_orthonormal_recurrence_values(z, α, β, degree)`](@ref).

The orthonormal polynomials satisfy

    √β_{k+1} p_{k+1}(z) = (z - α_k) p_k(z) - √β_k p_{k-1}(z),   p_0 = 1/√β_0

with the `k = 0` term `√β_0 p_{-1}` absent. Differentiating `m ≥ 1` times
w.r.t. `z` (Leibniz applied to the `(z - α_k)` factor) yields the derivative
recurrence

    √β_{k+1} p_{k+1}⁽ᵐ⁾(z) = m p_k⁽ᵐ⁻¹⁾(z) + (z - α_k) p_k⁽ᵐ⁾(z)
                              - √β_k p_{k-1}⁽ᵐ⁾(z)

with `p₀⁽ᵐ⁾ = 0` for `m > 0` and `pₖ⁽⁰⁾ = pₖ` (the plain recurrence
values).
All orders are computed in one pass over `m` (the `m`-th order reuses the
`(m-1)`-th), so the cost is `O(max_order × degree)` per point. Non-mutating
(each layer is a fresh broadcast) mirroring the AD-friendly style of
`_orthonormal_recurrence_values`.
"""
function _orthonormal_recurrence_derivative_values(
        z::AbstractVector{Tz},
        α::AbstractVector,
        β::AbstractVector,
        degree::Integer,
        max_order::Integer,
    ) where {Tz}
    N = length(z)
    D = Int(degree)
    M = Int(max_order)
    out = Array{Tz, 3}(undef, N, D + 1, M + 1)
    @inbounds for i in 1:N
        zi = z[i]

        # m = 0: plain forward recurrence (matches _orthonormal_recurrence_values).
        p0 = one(Tz) / sqrt(Tz(β[1]))
        out[i, 1, 1] = p0
        if D > 0
            out[i, 2, 1] = (zi - α[1]) * p0 / sqrt(Tz(β[2]))
            for k in 1:(D - 1)
                out[i, k + 2, 1] = ((zi - α[k + 1]) * out[i, k + 1, 1] - sqrt(Tz(β[k + 1])) * out[i, k, 1]) / sqrt(Tz(β[k + 2]))
            end
        end

        # m ≥ 1: differentiated recurrence; p₀⁽ᵐ⁾ = 0 seeds the sweep and the
        # k = 0 "√β₀ p_{-1}" term is absent, exactly as in the base recurrence.
        for m in 1:M
            Tm = Tz(m)
            out[i, 1, m + 1] = zero(Tz)
            if D > 0
                out[i, 2, m + 1] = (
                    Tm * out[i, 1, m] + (zi - α[1]) * out[i, 1, m + 1]
                ) / sqrt(Tz(β[2]))
                for k in 1:(D - 1)
                    out[i, k + 2, m + 1] = (
                        Tm * out[i, k + 1, m]
                        + (zi - α[k + 1]) * out[i, k + 1, m + 1]
                        - sqrt(Tz(β[k + 1])) * out[i, k, m + 1]
                    ) / sqrt(Tz(β[k + 2]))
                end
            end
        end
    end
    return out
end

"""
    _orthonormal_recurrence_derivative_order(z, α, β, degree, order) -> Matrix{Tz}

Evaluate only the `order`-th `z`-derivative of the orthonormal polynomials
`p_0 … p_degree` at every point of `z`, returning a `(length(z), degree + 1)`
matrix whose entry `[i, k + 1]` is `∂ᵒʳᵈᵉʳ p_k / ∂zᵒʳᵈᵉʳ` at `z[i]`.

Same differentiated three-term recurrence as
[`_orthonormal_recurrence_derivative_values`](@ref), but the sweep over
derivation order keeps only the current and previous `m`-layer instead of
materializing all `order + 1` of them, so peak memory is
`O(length(z) × degree)` rather than `O(length(z) × degree × order)`. Use this
when a single order is needed; use the full-tensor version when several orders
are consumed together (e.g. the reverse-mode pullback, which reads both the
values and the first derivative).
"""
function _orthonormal_recurrence_derivative_order(
        z::AbstractVector{Tz},
        α::AbstractVector,
        β::AbstractVector,
        degree::Integer,
        order::Integer,
    ) where {Tz}
    N = length(z)
    D = Int(degree)
    M = Int(order)
    out = Matrix{Tz}(undef, N, D + 1)

    # Two rolling layers of length D + 1: `cur` holds the m-th derivative row,
    # `prev` the (m-1)-th. Reused across points, so allocation is O(degree).
    cur = Vector{Tz}(undef, D + 1)
    prev = Vector{Tz}(undef, D + 1)
    @inbounds for i in 1:N
        zi = z[i]

        # m = 0: plain forward recurrence into `cur`.
        cur[1] = one(Tz) / sqrt(Tz(β[1]))
        if D > 0
            cur[2] = (zi - α[1]) * cur[1] / sqrt(Tz(β[2]))
            for k in 1:(D - 1)
                cur[k + 2] = ((zi - α[k + 1]) * cur[k + 1] - sqrt(Tz(β[k + 1])) * cur[k]) / sqrt(Tz(β[k + 2]))
            end
        end

        # m ≥ 1: differentiated recurrence; the m-th layer reuses the (m-1)-th,
        # so swap `cur`/`prev` and overwrite. p₀⁽ᵐ⁾ = 0 seeds each sweep and the
        # k = 0 "√β₀ p_{-1}" term is absent, exactly as in the base recurrence.
        for m in 1:M
            cur, prev = prev, cur
            Tm = Tz(m)
            cur[1] = zero(Tz)
            if D > 0
                cur[2] = (Tm * prev[1] + (zi - α[1]) * cur[1]) / sqrt(Tz(β[2]))
                for k in 1:(D - 1)
                    cur[k + 2] = (
                        Tm * prev[k + 1]
                        + (zi - α[k + 1]) * cur[k + 1]
                        - sqrt(Tz(β[k + 1])) * cur[k]
                    ) / sqrt(Tz(β[k + 2]))
                end
            end
        end

        for k in 1:(D + 1)
            out[i, k] = cur[k]
        end
    end
    return out
end

"""
    aPCE_PsiPolynomialMatrix_zygote(TrainingInput, MultivariatePolynomialDegrees, rb::RecurrenceCenteredBasis)

`RecurrenceCenteredBasis` dispatch: standardize `TrainingInput` per-dimension,
evaluate each dimension's orthonormal polynomials through the stable forward
three-term recurrence, then assemble the multivariate `Psi` product. Unlike the
monomial-coefficient (`CenteredBasis`) path this keeps every value O(1), so
orthonormality holds to high degree on skewed measures.
"""
function aPCE_PsiPolynomialMatrix_zygote(
        TrainingInput,
        MultivariatePolynomialDegrees,
        rb::RecurrenceCenteredBasis,
    )
    μ_row = reshape(rb.μ, 1, :)
    σ_row = reshape(rb.σ, 1, :)
    z = (TrainingInput .- μ_row) ./ σ_row

    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(z, 1)

    # Per-dimension orthonormal values: Pvals[dim][point, degree + 1].
    Pvals = [
        _orthonormal_recurrence_values(
                z[:, dim], @view(rb.α[:, dim]), @view(rb.β[:, dim]), rb.degree,
            )
            for dim in 1:InputDimensions
    ]

    Psi = [
        prod(
            Pvals[dim][j, MultivariatePolynomialDegrees[i, dim] + 1]
                for dim in 1:InputDimensions
        )
            for i in 1:NumberOfTerms, j in 1:NCpoints
    ]
    return reshape(Psi, NumberOfTerms, NCpoints)
end

# `aPCE_PsiPolynomialMatrix` (the non-`_zygote` name used by the high-level
# aPCE/predict/train! path) shares the recurrence evaluator — the assembly is
# identical and already AD-friendly.
function aPCE_PsiPolynomialMatrix(
        TrainingInput,
        MultivariatePolynomialDegrees,
        rb::RecurrenceCenteredBasis,
    )
    return aPCE_PsiPolynomialMatrix_zygote(TrainingInput, MultivariatePolynomialDegrees, rb)
end

# NOTE: `aPCE_DerivativeBasis` (the public physical-space derivative of the Psi
# matrix) lives in APCEhighlevel.jl with the other `aPCE`-struct-typed methods,
# because the `aPCE` struct is defined there (APCEfunctions.jl is included
# first and cannot reference the type in a signature). It dispatches on the
# `OrthonormalBasis` backend and calls `_orthonormal_recurrence_derivative_values`
# (defined above) for the `RecurrenceCenteredBasis` path.

"""
    _recurrence_nodes(rb, dim, n) -> Vector

Golub–Welsch nodes for dimension `dim`: the `n` roots of the orthonormal
polynomial `p_n` are the eigenvalues of the symmetric-tridiagonal Jacobi matrix
(diagonal `α₀ … α_{n-1}`, off-diagonal `√β₁ … √β_{n-1}`), mapped back from
z-space to x-space via `μ + σ·z`. More stable than root-finding on monomial
coefficients.
"""
function _recurrence_nodes(rb::RecurrenceCenteredBasis{T}, dim::Integer, n::Integer) where {T}
    diagonal = collect(@view rb.α[1:n, dim])
    off = sqrt.(max.(zero(T), collect(@view rb.β[2:n, dim])))
    z = LinearAlgebra.eigvals(LinearAlgebra.SymTridiagonal(diagonal, off))
    return rb.μ[dim] .+ rb.σ[dim] .* z
end

# Nodes of the top (degree + 1) polynomial per basis backend, used by
# `GaussianCollocation`. Dispatch keeps each backend's path fully type-stable.
function _basis_nodes(basis::AbstractArray, dim::Integer, degree::Integer, ::Type{T}) where {T}
    return real.(PolynomialRoots.roots(basis[degree + 2, :, dim]))
end
function _basis_nodes(rb::RecurrenceCenteredBasis, dim::Integer, degree::Integer, ::Type{T}) where {T}
    return _recurrence_nodes(rb, dim, degree + 1)
end

# The public API exports `PsiPolynomialMatrix_zygote` (see the module's `export`
# list), but the implementation above is named `aPCE_PsiPolynomialMatrix_zygote`.
# Without this alias the exported name is unbound, so any downstream `using` +
# call raises UndefVarError and Aqua's undefined_exports check fails. Aliasing
# (rather than renaming) keeps the public surface byte-for-byte identical while
# making the documented public name actually callable.
const PsiPolynomialMatrix_zygote = aPCE_PsiPolynomialMatrix_zygote

"""
    aPCE_PsiPolynomialMatrix(TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)

Compute the Psi matrix optimized for AD compatibility.
Uses scalar polynomial evaluation to avoid intermediate array allocations.
This is AD-friendly: no in-place mutations, just scalar operations building up Psi.
"""
function aPCE_PsiPolynomialMatrix(
        TrainingInput::AbstractArray{T},
        MultivariatePolynomialDegrees,
        OrthonormalBasis::AbstractArray{S},
    ) where {S, T <: Real}
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = ones(T, NumberOfTerms, NCpoints)

    @inbounds for i in 1:NumberOfTerms
        for ii in 1:InputDimensions
            degree = MultivariatePolynomialDegrees[i, ii] + 1
            coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
            # Scalar evaluation for each point - no intermediate array allocation
            for j in 1:NCpoints
                xj = TrainingInput[j, ii]
                poly_val = evaluate_polynomial_horner_scalar(xj, coeffs)
                Psi[i, j] *= poly_val
            end
        end
    end
    return Psi
end

# ===== MOMENT COMPUTATION =====

"""
   compute_moments!(m, Data, NumberOfDataPoints, dd)

Compute moments of data up to order 2*dd+1 in-place.
"""
@inline function compute_moments!(
        m::AbstractArray{T},
        Data::AbstractArray{S},
        NumberOfDataPoints::Integer,
        dd::Integer,
    ) where {T <: Real, S <: Real}
    current_power = Vector{T}(undef, length(Data))
    fill!(current_power, one(T))
    for l in 0:(2 * dd + 1)
        m[l + 1] = sum(current_power) / NumberOfDataPoints
        current_power .*= Data
    end
    return nothing
end

# ===== COLLOCATION POINTS =====

"""
   GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis, InputDistribution, NumberOfTerms; strategy = :PCM)

Generate Gaussian collocation points for quadrature.
- `strategy`: :PCM (Probabilistic Collocation Method) or :FT (Full Tensor)
"""
function GaussianCollocation(
        input_dimensions, ExpansionDegree, OrthonormalBasis::AbstractArray{T},
        InputDistribution::AbstractArray{T}, NumberOfTerms; strategy = :PCM,
    )::Matrix{T} where {T <: Real}

    polynomial_roots = zeros(T, input_dimensions, ExpansionDegree + 1)
    @inbounds for d in Base.oneto(Int64(input_dimensions))
        polynomial_basis = @views OrthonormalBasis[:, :, d]
        roots = PolynomialRoots.roots(@views polynomial_basis[ExpansionDegree + 1, :])
        real_roots = real.(roots)
        polynomial_roots[d, 1:length(real_roots)] = real_roots
    end

    PointsVector = 1:(ExpansionDegree + 1) |> collect
    UniqueCombinations = reduce(
        vcat,
        [collect(t)' for t in Iterators.product([PointsVector for _ in 1:input_dimensions]...)],
    )

    sort_indices = sortperm(sum(UniqueCombinations; dims = 2); dims = 1)
    SortUniqueCombinations = UniqueCombinations[sort_indices[:], :]

    if strategy == :FT
        return Matrix{T}(SortUniqueCombinations)
    elseif strategy == :PCM
        temp = abs.(polynomial_roots .- StatsBase.mean(InputDistribution; dims = 1)[:, :][1])
        temp_sort = mapslices(sortperm, temp, dims = 2)
        @inbounds for i in axes(polynomial_roots, 1)
            polynomial_roots[i, :] = @views polynomial_roots[i, temp_sort[i, :]]
        end
        collocation_points = zeros(T, NumberOfTerms, input_dimensions)
        @inbounds for i in 1:NumberOfTerms
            for j in axes(SortUniqueCombinations, 2)
                collocation_points[i, j] =
                    @views polynomial_roots[j, Int(SortUniqueCombinations[i, j])]
            end
        end
        collocation_points = sortslices(collocation_points, dims = 1, by = x -> x[1])
        return collocation_points
    end
end

@testitem "GaussianCollocation_iterators_product_equivalence" begin
    using Random, StatsBase
    # Verify the Iterators.product refactor produces a correct tensor grid
    # for small dimensions where the result is easy to reason about.
    for (dims, deg) in [(2, 2), (3, 2), (3, 3), (4, 2)]
        x = randn(200, dims)
        basis = create_basis(x, deg; center_data = false)
        n_terms = binomial(dims + deg, deg)
        pts = GaussianCollocation(dims, deg, basis, x, n_terms; strategy = :PCM)

        # Shape: (n_terms, dims)
        @test size(pts) == (n_terms, dims)
        # No NaN / Inf
        @test all(!isnan, pts)
        @test all(!isinf, pts)
        # Each column should contain only roots of the 1D polynomial (degree+1 unique values)
        for j in 1:dims
            @test length(unique(round.(pts[:, j]; digits = 8))) <= deg + 1
        end
    end
end

# ===== EVALUATION FUNCTIONS =====

"""
   compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)

Compose the Psi matrix for evaluation.
"""
function compose_Ψ(
        x::AbstractArray{T},
        MultivariatePolynomialDegrees,
        OrthonormalBasis,
        degree,
    ) where {T}
    Ψ =
        aPCE_PsiPolynomialMatrix_zygote(
        x,
        MultivariatePolynomialDegrees,
        OrthonormalBasis,
    )' |> Matrix{T}
    return Ψ
end

"""
   evaluate_Ψ(x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)

Evaluate the polynomial expansion at points x with given coefficients.
"""
function evaluate_Ψ(
        x,
        coeffs,
        MultivariatePolynomialDegrees,
        OrthonormalBasis,
        degree,
        name,
    )
    T = eltype(coeffs)
    Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
    # Replaced @tensor with explicit matrix multiplication for Mooncake AD compatibility
    # Note: Ψ[k,i] * coeffs[i,j] sums over i, giving (Ψ * coeffs)[k,j]
    PredictionOutput = Ψ * coeffs
    return PredictionOutput
end

"""
   coeffs_from_basis(OrthonormalBasis, degree, ii)

Extract coefficients for a specific degree and dimension from the basis.
"""
function coeffs_from_basis(OrthonormalBasis, degree, ii)
    return OrthonormalBasis[degree, 1:degree, ii]
end

# ===== TRAINING FUNCTIONS =====

"""
   train(Ψ, y_rhs; bayesian_inversion = :true, reg_order = 0)

Train the polynomial expansion to find optimal coefficients.
- `Ψ`: Design matrix
- `y_rhs`: Target values
- `bayesian_inversion`: Whether to use Bayesian regularization
- `reg_order`: Regularization order
"""
function train(
        Ψ::AbstractArray{T},
        y_rhs;
        bayesian_inversion = :true,
        reg_order = 0,
    ) where {T <: Real}
    NumberOfTerms = size(Ψ, 2)
    output_dimensions = size(y_rhs, 2)
    coeffs = zeros(T, NumberOfTerms, output_dimensions)

    # Compute pseudo-inverse
    Psi_inv = pinv(Ψ; rtol = sqrt(eps(real(float(oneunit(eltype(Ψ)))))))
    @einsum coeffs[i, k] = Psi_inv[i, j] * y_rhs[j, k]

    if bayesian_inversion
        @info "Using bayesian regularization to find the expansion coefficients"
        x₀ = coeffs
        for i in axes(y_rhs, 2)
            @info "Bayesian regularization for axis $i"
            coeffs[:, i] .= invert(
                Ψ, y_rhs[:, i], Lₖx₀(reg_order, view(x₀, :, i));
                alg = :gcv_svd,
                method = LBFGS(linesearch = LineSearches.BackTracking()),
            )
        end
    end

    ChainRulesCore.ignore_derivatives() do
        for k in axes(coeffs, 2)
            res = (@views sqrt(mean((Ψ * coeffs[:, k] .- y_rhs[:, k]) .^ 2)))
            @info "Error for axis $k" res
        end
    end
    return coeffs
end


# ===== TESTS =====


@testitem "solve_levenberg_marquardt_basic_and_illconditioned" begin
    using LinearAlgebra
    using ErrorTypes
    # Well-conditioned system
    A = [3.0 2.0; 1.0 2.0]
    b = [5.0, 3.0]
    x_true = [1.0, 1.0]
    x_refined = unwrap(ArbitraryPolynomialChaosExpansion.solve_levenberg_marquardt(A, b))
    @test isapprox(x_refined, x_true, atol = 1.0e-10)

    eps_val = 1.0e-8
    A_ill = [1.0 1.0; 1.0 1.0 + eps_val]
    b_ill = [2.0, 2.0 + eps_val]
    x_true_ill = [1.0, 1.0]
    x_refined_ill =
        unwrap(ArbitraryPolynomialChaosExpansion.solve_levenberg_marquardt(A_ill, b_ill))
    @test isapprox(x_refined_ill, x_true_ill, atol = 1.0e-6)

    x_single = A_ill \ b_ill
    err_single = norm(x_single - x_true_ill)
    err_refined = norm(x_refined_ill - x_true_ill)
    @test err_refined <= err_single + 1.0e-6
end

####

@testitem "aPCE_MultivariatePolynomialDegrees" begin
    @test aPCE_MultivariatePolynomialDegrees(2, 1, 1.0, 1.0) == [0 0; 0 1; 1 0]
    @test aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0) ==
        [0 0; 0 1; 1 0; 0 2; 1 1; 2 0]
    @inferred aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0)
end

@testitem "aPCE_OrthonormalBasis" begin
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true)) ≈
        [1.0 0.0; -0.5 1.5]
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false)) ≈
        [1.0 0.0; -0.5 1.5]
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false))
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true))
end

@testitem "numberPolynomials" begin
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(3, 2) == 10
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(5, 3) == 56
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(0, 0) == 1
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(1, 1) == 2
    @test typeof(ArbitraryPolynomialChaosExpansion.numberPolynomials(3, 2)) == Int
end

@testitem "aPCE_MultivariatePolynomialDegrees_test" begin
    @test aPCE_MultivariatePolynomialDegrees(2, 1, 1.0, 1.0) == [0 0; 0 1; 1 0]
    @test aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0) ==
        [0 0; 0 1; 1 0; 0 2; 1 1; 2 0]
    @inferred aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0)
end


@testitem "aPCE_OrthonormalBasis_test" begin
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true)) ≈
        [1.0 0.0; -0.5 1.5]
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false)) ≈
        [1.0 0.0; -0.5 1.5]
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false))
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true))
end

@testitem "numberPolynomials_test" begin
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(3, 2) == 10
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(5, 3) == 56
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(0, 0) == 1
    @test ArbitraryPolynomialChaosExpansion.numberPolynomials(1, 1) == 2
    @test typeof(ArbitraryPolynomialChaosExpansion.numberPolynomials(3, 2)) == Int
end

@testitem "type_stability_tests" begin
    # Test type stability of normalization_functions
    x = rand(10, 2)
    normalize, inverse_normalize =
        ArbitraryPolynomialChaosExpansion.normalization_functions(x)
    @inferred ArbitraryPolynomialChaosExpansion.normalization_functions(x)
    @test typeof(normalize(x)) == typeof(x)
    @test typeof(inverse_normalize(x)) == typeof(x)

    # Test type stability of compute_Psi_element
    TrainingInput = rand(10, 2)
    MultivariatePolynomialDegrees = [0 0; 0 1; 1 0]
    OrthonormalBasis = rand(3, 3, 2)
    @inferred ArbitraryPolynomialChaosExpansion.compute_Psi_element(
        1,
        1,
        TrainingInput,
        MultivariatePolynomialDegrees,
        OrthonormalBasis,
        2,
    )

    # Test type stability of evalpoly_two
    x = 2.0
    coeffs = [1.0, 2.0, 3.0]
    @inferred ArbitraryPolynomialChaosExpansion.evalpoly_two(x, coeffs)
    @test typeof(ArbitraryPolynomialChaosExpansion.evalpoly_two(x, coeffs)) == Float64

    # Test type stability of evaluate_derivative_horner
    @inferred ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(x, coeffs)
    @test typeof(ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(x, coeffs)) ==
        Float64

    # Test type stability of evaluate_polynomial_horner_array
    x_array = [1.0, 2.0, 3.0]
    @inferred ArbitraryPolynomialChaosExpansion.evaluate_polynomial_horner_array(
        x_array,
        coeffs,
    )
    @test typeof(
        ArbitraryPolynomialChaosExpansion.evaluate_polynomial_horner_array(x_array, coeffs),
    ) == Vector{Float64}

    # Test type stability of train
    Ψ = rand(10, 5)
    y_rhs = rand(10, 2)
    @inferred ArbitraryPolynomialChaosExpansion.train(Ψ, y_rhs)
    @test typeof(ArbitraryPolynomialChaosExpansion.train(Ψ, y_rhs)) == Matrix{Float64}

    # Test type stability of aPCE_FullBasis
    Data = rand(10)
    Degree = 2
    @inferred ArbitraryPolynomialChaosExpansion.aPCE_FullBasis(Data, Degree)
    @test typeof(ArbitraryPolynomialChaosExpansion.aPCE_FullBasis(Data, Degree)) ==
        Matrix{Float64}

    # Test type stability of GaussianCollocation
    input_dimensions = 2
    ExpansionDegree = 2
    OrthonormalBasis = rand(3, 3, 2)
    InputDistribution = rand(10, 2)
    NumberOfTerms = 6
    @inferred ArbitraryPolynomialChaosExpansion.GaussianCollocation(
        input_dimensions,
        ExpansionDegree,
        OrthonormalBasis,
        InputDistribution,
        NumberOfTerms,
    )
    @test typeof(
        ArbitraryPolynomialChaosExpansion.GaussianCollocation(
            input_dimensions,
            ExpansionDegree,
            OrthonormalBasis,
            InputDistribution,
            NumberOfTerms,
        ),
    ) == Matrix{Float64}
end

@testitem "edge_cases_tests" begin
    # Test edge cases for evalpoly_two
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(0.0, [1.0]) == 1.0  # Constant polynomial
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(1.0, [0.0, 0.0]) == 0.0  # Zero polynomial
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(Inf, [1.0, 2.0]) == Inf  # Infinity input
    @test isnan(ArbitraryPolynomialChaosExpansion.evalpoly_two(NaN, [1.0, 2.0]))  # NaN input

    # Test edge cases for evaluate_derivative_horner
    @test ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(0.0, [1.0]) == 0.0  # Constant polynomial
    @test ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(1.0, [0.0, 0.0]) ==
        0.0  # Zero polynomial

    # Test edge cases for train
    Ψ = zeros(5, 3)
    y_rhs = zeros(5, 2)
    @test all(iszero, ArbitraryPolynomialChaosExpansion.train(Ψ, y_rhs))  # Zero inputs
    @test size(ArbitraryPolynomialChaosExpansion.train(Ψ, y_rhs)) == (3, 2)  # Correct output size

    # Test edge cases for aPCE_FullBasis
    @test size(ArbitraryPolynomialChaosExpansion.aPCE_FullBasis([1.0], 0)) == (1, 1)  # Degree 0
    @test size(ArbitraryPolynomialChaosExpansion.aPCE_FullBasis([1.0], 1)) == (2, 2)  # Degree 1
end


@testitem "aPCE_OrthonormalBasis1" begin
    @test ArbitraryPolynomialChaosExpansion.aPCE_OrthonormalBasis(
        [1 / sqrt(3), -1 / sqrt(3), 1.0],
        1,
        Val(true),
    ) ≈ [1.0 0.0; -0.5 1.5]
    @test ArbitraryPolynomialChaosExpansion.aPCE_OrthonormalBasis(
        [1 / sqrt(3), -1 / sqrt(3), 1.0],
        1,
        Val(false),
    ) ≈ [1.0 0.0; -0.5 1.5]
    @inferred ArbitraryPolynomialChaosExpansion.aPCE_OrthonormalBasis(
        [1 / sqrt(3), -1 / sqrt(3), 1.0],
        1,
        Val(false),
    )
    @inferred ArbitraryPolynomialChaosExpansion.aPCE_OrthonormalBasis(
        [1 / sqrt(3), -1 / sqrt(3), 1.0],
        1,
        Val(true),
    )
end

@testitem "aPCE_OrthonormalBasis_comprehensive_test_true" begin
    using Statistics: mean, std
    # Test with various input types and sizes
    T = Float64
    Data = rand(T, 1000)
    Degree = 3

    # Test basic functionality
    basis = aPCE_OrthonormalBasis(Data, Degree, Val(true))
    @test size(basis) == (Degree + 1, Degree + 1)
    @test all(!isnan, basis)
    @test all(!isinf, basis)

    # Test type stability
    @inferred aPCE_OrthonormalBasis(Data, Degree, Val(true))

    # Test orthonormality
    for i in 1:(Degree + 1)
        for j in 1:(Degree + 1)
            # Evaluate both basis functions at all data points
            poly_i = zeros(T, length(Data))
            poly_j = zeros(T, length(Data))
            for k in 1:length(Data)
                for d in 0:(i - 1)
                    poly_i[k] += basis[i, d + 1] * Data[k]^d
                end
                for d in 0:(j - 1)
                    poly_j[k] += basis[j, d + 1] * Data[k]^d
                end
            end
            # Compute dot product (average of product over data points)
            dot_product = sum(poly_i .* poly_j) / length(Data)
            if i == j
                @test isapprox(dot_product, one(T), rtol = 1.0e-5)
            else
                @test isapprox(dot_product, zero(T), atol = 1.0e-10)
            end
        end
    end
end

@testitem "aPCE_OrthonormalBasis_comprehensive_test_false" begin
    using Statistics: mean, std
    # Test with various input types and sizes
    T = Float64
    Data = rand(T, 1000)
    Degree = 3

    # Test basic functionality
    basis = aPCE_OrthonormalBasis(Data, Degree, Val(false))
    @test size(basis) == (Degree + 1, Degree + 1)
    @test all(!isnan, basis)
    @test all(!isinf, basis)

    # Test type stability
    @inferred aPCE_OrthonormalBasis(Data, Degree, Val(false))

    # Test orthonormality
    for i in 1:(Degree + 1)
        for j in 1:(Degree + 1)
            # Evaluate both basis functions at all data points
            poly_i = zeros(T, length(Data))
            poly_j = zeros(T, length(Data))
            for k in 1:length(Data)
                for d in 0:(i - 1)
                    poly_i[k] += basis[i, d + 1] * Data[k]^d
                end
                for d in 0:(j - 1)
                    poly_j[k] += basis[j, d + 1] * Data[k]^d
                end
            end
            # Compute dot product (average of product over data points)
            dot_product = sum(poly_i .* poly_j) / length(Data)
            if i == j
                @test isapprox(dot_product, one(T), rtol = 1.0e-5)
            else
                @test isapprox(dot_product, zero(T), atol = 1.0e-5)
            end
        end
    end
end


@testitem "reverse_columns_test" begin
    mat1 = [1 2 3; 4 5 6; 7 8 9]
    expected1 = [3 2 1; 6 5 4; 9 8 7]
    ArbitraryPolynomialChaosExpansion.reverse_columns!(mat1)
    @test mat1 == expected1

    mat2 = [1 2; 3 4; 5 6]
    expected2 = [2 1; 4 3; 6 5]
    ArbitraryPolynomialChaosExpansion.reverse_columns!(mat2)
    @test mat2 == expected2

    mat3 = [1 2 3 4; 5 6 7 8]
    expected3 = [4 3 2 1; 8 7 6 5]
    ArbitraryPolynomialChaosExpansion.reverse_columns!(mat3)
    @test mat3 == expected3

    @test typeof(ArbitraryPolynomialChaosExpansion.reverse_columns!(mat1)) == Matrix{Int}
end


@testitem "evaluate_derivative_horner_test" begin
    @test ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(
        2.0,
        [1.0, 2.0, 3.0],
    ) == 14.0  # Derivative of 1 + 2x + 3x^2 at x=2
    @test ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(
        0.0,
        [1.0, 2.0, 3.0],
    ) == 2.0   # Derivative of 1 + 2x + 3x^2 at x=0
    @test ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(
        1.0,
        [0.0, 0.0, 0.0],
    ) == 0.0   # Derivative of 0 polynomial at x=1
    @test ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(1.0, [5.0]) == 0.0             # Derivative of constant polynomial at x=1
    @test typeof(
        ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(2.0, [1.0, 2.0, 3.0]),
    ) == Float64
    @inferred ArbitraryPolynomialChaosExpansion.evaluate_derivative_horner(1.0, [5.0])
end

@testitem "evalpoly_two_test" begin
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(2.0, [1.0, 2.0, 3.0, 4.0]) == 49.0  # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=2
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(0.0, [1.0, 2.0, 3.0, 4.0]) == 1.0   # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=0
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(1.0, [0.0, 0.0, 0.0, 0.0]) == 0.0   # Zero polynomial at x=1
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(1.0, [5.0]) == 5.0                  # Constant polynomial at x=1
    @test ArbitraryPolynomialChaosExpansion.evalpoly_two(2.0, [1.0, -1.0, 1.0, -1.0]) ==
        -5.0  # Polynomial 1 - x + x^2 - x^3 at x=2
end

@testitem "EstrinPoly_test" begin
    using Estrin
    @test Estrin.Poly([1.0, 2.0, 3.0, 4.0]).(2.0) == 49.0  # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=2
    @test Estrin.Poly([1.0, 2.0, 3.0, 4.0]).(0.0) == 1.0   # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=0
    @test Estrin.Poly([0.0, 0.0, 0.0, 0.0]).(1.0) == 0.0   # Zero polynomial at x=1
    @test Estrin.Poly([5.0]).(1.0) == 5.0                  # Constant polynomial at x=1
    @test Estrin.Poly([1.0, -1.0, 1.0, -1.0]).(2.0) == -5.0  # Polynomial 1 - x + x^2 - x^3 at x=2
end


@testitem "derivative_coeffs_test" begin
    @test ArbitraryPolynomialChaosExpansion.derivative_coeffs([1.0, 2.0, 3.0]) == [2.0, 6.0]  # Derivative of 1 + 2x + 3x^2
    @test ArbitraryPolynomialChaosExpansion.derivative_coeffs([0.0, 0.0, 0.0]) == [0.0, 0.0]  # Derivative of 0 polynomial
    @test ArbitraryPolynomialChaosExpansion.derivative_coeffs([5.0]) == [0.0]                 # Derivative of constant polynomial
    @test ArbitraryPolynomialChaosExpansion.derivative_coeffs([1.0, -1.0, 1.0, -1.0]) ==
        [-1.0, 2.0, -3.0]  # Derivative of 1 - x + x^2 - x^3
end

@testitem "create_basis_orthonormal_true_test" begin
    # Test basic functionality
    x = rand(100, 2)
    degree = 3
    basis = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(true))

    @test size(basis) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis) == eltype(x)
    @test all(!isnan, basis)
    @test all(!isinf, basis)

    # Test type stability
    @inferred ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(true))

    # Test with different input types
    x_float32 = rand(Float32, 50, 3)
    basis_float32 = ArbitraryPolynomialChaosExpansion.create_basis(x_float32, 2, Val(true))
    @test eltype(basis_float32) == Float32
    @test size(basis_float32) == (3, 3, 3)

    # Test edge cases
    x_single = rand(10, 1)
    basis_single = ArbitraryPolynomialChaosExpansion.create_basis(x_single, 0, Val(true))
    @test size(basis_single) == (1, 1, 1)
    @test basis_single[1, 1, 1] ≈ 1.0 atol = 1.0e-10
end

@testitem "create_basis_orthonormal_false_test" begin
    # Test basic functionality
    x = rand(100, 2)
    degree = 3
    basis = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(false))

    @test size(basis) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis) == eltype(x)
    @test all(!isnan, basis)
    @test all(!isinf, basis)

    # Test type stability
    @inferred ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(false))

    # Test that it produces the expected full basis structure
    # For degree 3, each dimension should have a (4,4) matrix with upper triangular structure
    for dim in 1:size(x, 2)
        basis_dim = basis[:, :, dim]
        # Check upper triangular structure (1s on and above diagonal, 0s below)
        for i in 1:(degree + 1)
            for j in 1:(degree + 1)
                if i >= j
                    @test basis_dim[i, j] == 1.0
                else
                    @test basis_dim[i, j] == 0.0
                end
            end
        end
    end

    # Test with different input types
    x_float32 = rand(Float32, 50, 3)
    basis_float32 = ArbitraryPolynomialChaosExpansion.create_basis(x_float32, 2, Val(false))
    @test eltype(basis_float32) == Float32
    @test size(basis_float32) == (3, 3, 3)

    # Test edge cases
    x_single = rand(10, 1)
    basis_single = ArbitraryPolynomialChaosExpansion.create_basis(x_single, 0, Val(false))
    @test size(basis_single) == (1, 1, 1)
    @test basis_single[1, 1, 1] == 1.0

    # Test degree 1 case
    basis_degree1 = ArbitraryPolynomialChaosExpansion.create_basis(x_single, 1, Val(false))
    @test size(basis_degree1) == (2, 2, 1)
    @test basis_degree1[:, :, 1] == [1.0 0.0; 1.0 1.0]
end

@testitem "create_basis_comparison_test" begin
    # Test that orthonormal and non-orthonormal bases have same structure but different values
    x = rand(50, 2)
    degree = 2

    basis_ortho = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(true))
    basis_full = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(false))

    @test size(basis_ortho) == size(basis_full)
    @test eltype(basis_ortho) == eltype(basis_full)

    # Orthonormal basis should have different values than full basis
    @test !isapprox(basis_ortho, basis_full, atol = 1.0e-10)

    # Full basis should have the expected structure (upper triangular with 1s)
    for dim in 1:size(x, 2)
        basis_dim = basis_full[:, :, dim]
        for i in 1:(degree + 1)
            for j in 1:(degree + 1)
                if i >= j
                    @test basis_dim[i, j] == 1.0
                else
                    @test basis_dim[i, j] == 0.0
                end
            end
        end
    end

    # Orthonormal basis should not have this simple structure
    for dim in 1:size(x, 2)
        basis_dim = basis_ortho[:, :, dim]
        # Check that it's not the identity matrix (which would be the case for full basis)
        @test !all(
            basis_dim[i, j] == (i >= j ? 1.0 : 0.0) for i in 1:(degree + 1), j in 1:(degree + 1)
        )
    end
end

@testitem "create_basis_functional_test" begin
    # Test that the bases can be used in polynomial evaluation
    x = rand(20, 2)
    degree = 2

    # Create both types of bases
    basis_ortho = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(true))
    basis_full = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(false))

    # Test that we can extract coefficients from both bases
    for dim in 1:size(x, 2)
        for d in 1:(degree + 1)
            coeffs_ortho =
                ArbitraryPolynomialChaosExpansion.coeffs_from_basis(basis_ortho, d, dim)
            coeffs_full =
                ArbitraryPolynomialChaosExpansion.coeffs_from_basis(basis_full, d, dim)

            @test length(coeffs_ortho) == d
            @test length(coeffs_full) == d
            @test all(!isnan, coeffs_ortho)
            @test all(!isnan, coeffs_full)
        end
    end

    # Test that polynomial evaluation works with both bases
    test_point = rand(2)
    multivar_degrees = [0 0; 0 1; 1 0; 0 2; 1 1; 2 0]  # 2D, degree 2

    # This should work without errors for both basis types
    @test_nowarn ArbitraryPolynomialChaosExpansion.aPCE_PsiPolynomialMatrix_zygote(
        test_point',
        multivar_degrees,
        basis_ortho,
    )
    @test_nowarn ArbitraryPolynomialChaosExpansion.aPCE_PsiPolynomialMatrix_zygote(
        test_point',
        multivar_degrees,
        basis_full,
    )
end

@testitem "create_basis_default_test" begin
    using LinearAlgebra
    using StatsBase
    # Test the default create_basis function (should default to orthonormal)
    x = rand(30, 2)
    degree = 2

    # Test default behavior
    basis_default = ArbitraryPolynomialChaosExpansion.create_basis(x, degree)
    basis_ortho = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(true))

    # Default should be the same as orthonormal
    @test isapprox(basis_default, basis_ortho, atol = 1.0e-10)

    # Test with center_data parameter
    basis_default_centered =
        ArbitraryPolynomialChaosExpansion.create_basis(x, degree, center_data = true)
    basis_default_uncentered =
        ArbitraryPolynomialChaosExpansion.create_basis(x, degree, center_data = false)

    # Debug: Check the data properties
    @info "Data mean" StatsBase.mean(x, dims = 1)
    @info "Data std" StatsBase.std(x, dims = 1)
    @info "Data range" [minimum(x, dims = 1) maximum(x, dims = 1)]

    # Check if the difference is actually significant
    diff_norm = norm(basis_default_centered - basis_default_uncentered)
    @info "Difference norm" diff_norm
    @info "Relative difference" diff_norm / norm(basis_default_centered)

    @info "basis_default_centered" basis_default_centered
    @info "basis_default_uncentered" basis_default_uncentered

    # The backward transformation in the centered version should make the results nearly identical
    # Centering is for numerical stability, not to change the final result
    @test isapprox(basis_default_centered, basis_default_uncentered, atol = 1.0e-8)

    # Test type stability
    @inferred ArbitraryPolynomialChaosExpansion.create_basis(x, degree)
    @inferred ArbitraryPolynomialChaosExpansion.create_basis(x, degree, center_data = true)
    @inferred ArbitraryPolynomialChaosExpansion.create_basis(x, degree, center_data = false)
end

@testitem "create_basis_centered_vs_uncentered_test" begin
    using LinearAlgebra
    using StatsBase
    # Test with data that has a clear mean different from 1.0
    # This should make the difference between centered and uncentered modes obvious
    x = [1.0, 5.0, 10.0, 15.0, 20.0] .* ones(5, 2)  # Data with mean 10.2
    degree = 2

    @info "Test data mean" StatsBase.mean(x, dims = 1)
    @info "Test data std" StatsBase.std(x, dims = 1)

    basis_centered =
        ArbitraryPolynomialChaosExpansion.create_basis(x, degree, center_data = true)
    basis_uncentered =
        ArbitraryPolynomialChaosExpansion.create_basis(x, degree, center_data = false)

    # These should be nearly identical due to the backward transformation
    diff_norm = norm(basis_centered - basis_uncentered)
    @info "Difference norm for test data" diff_norm
    @info "Relative difference for test data" diff_norm / norm(basis_centered)

    # The backward transformation should make them nearly identical
    @test isapprox(basis_centered, basis_uncentered, atol = 1.0e-8)

    # Test that both bases are valid (no NaN or Inf)
    @test all(!isnan, basis_centered)
    @test all(!isnan, basis_uncentered)
    @test all(!isinf, basis_centered)
    @test all(!isinf, basis_uncentered)
end

@testitem "create_basis_centering_effect_test" begin
    using LinearAlgebra
    using StatsBase
    # Test to understand the centering effect
    x = [1.0, 5.0, 10.0, 15.0, 20.0] .* ones(5, 2)  # Data with mean 10.2
    degree = 2

    # Test individual aPCE_OrthonormalBasis calls to see the effect
    x_col1 = view(x, :, 1)
    mean_x = StatsBase.mean(x_col1)
    @info "Data mean" mean_x

    basis_centered = aPCE_OrthonormalBasis(x_col1, degree, Val(true))
    basis_uncentered = aPCE_OrthonormalBasis(x_col1, degree, Val(false))

    @info "Centered basis" basis_centered
    @info "Uncentered basis" basis_uncentered

    # Check if they're nearly identical (which would indicate the backward transformation is canceling the effect)
    diff_norm = norm(basis_centered - basis_uncentered)
    @info "Difference norm" diff_norm
    @info "Relative difference" diff_norm / norm(basis_centered)

    # If the backward transformation is working as intended, these should be nearly identical
    # This suggests that centering is for numerical stability, not to change the final result
    @test isapprox(basis_centered, basis_uncentered, atol = 1.0e-8)
end

@testitem "create_basis_numerical_stability_test" begin
    # Test that centering provides numerical stability benefits
    # Use data with very large values that could cause numerical issues
    x_large = [1.0e10, 2.0e10, 3.0e10, 4.0e10, 5.0e10] .* ones(5, 2)
    degree = 2

    # Both should work without numerical errors
    basis_centered =
        ArbitraryPolynomialChaosExpansion.create_basis(x_large, degree, center_data = true)
    basis_uncentered =
        ArbitraryPolynomialChaosExpansion.create_basis(x_large, degree, center_data = false)

    # Both should be valid
    @test all(!isnan, basis_centered)
    @test all(!isnan, basis_uncentered)
    @test all(!isinf, basis_centered)
    @test all(!isinf, basis_uncentered)

    # They should be nearly identical
    @test isapprox(basis_centered, basis_uncentered, atol = 1.0e-6)

end


@testitem "create_basis_dispatch_tests" begin
    # Setup test data
    x = rand(50, 2)
    degree = 2

    # Test 1: Default - orthonormal with centering
    basis1 = create_basis(x, degree)
    @test size(basis1) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis1) == eltype(x)
    @test all(!isnan, basis1)
    @test all(!isinf, basis1)
    @inferred create_basis(x, degree)
end

@testitem "create_basis_explicit_orthonormal_with_centering" begin
    x = rand(50, 2)
    degree = 2

    # Test 2: Explicit orthonormal with centering
    basis2 = create_basis(x, degree, Val(true); center_data = true)
    @test size(basis2) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis2) == eltype(x)
    @test all(!isnan, basis2)
    @test all(!isinf, basis2)
    @inferred create_basis(x, degree, Val(true); center_data = true)
end

@testitem "create_basis_orthonormal_without_centering" begin
    x = rand(50, 2)
    degree = 2

    # Test 3: Orthonormal without centering
    basis3 = create_basis(x, degree, Val(true); center_data = false)
    @test size(basis3) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis3) == eltype(x)
    @test all(!isnan, basis3)
    @test all(!isinf, basis3)
    @inferred create_basis(x, degree, Val(true); center_data = false)
end

@testitem "create_basis_monomial_basis" begin
    x = rand(50, 2)
    degree = 2

    # Test 4: Monomial basis (center_data ignored)
    basis4 = create_basis(x, degree, Val(false))
    @test size(basis4) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis4) == eltype(x)

    # Check monomial basis structure (upper triangular with 1s)
    for dim in 1:size(x, 2)
        basis_dim = basis4[:, :, dim]
        for i in 1:(degree + 1)
            for j in 1:(degree + 1)
                if i >= j
                    @test basis_dim[i, j] == 1.0
                else
                    @test basis_dim[i, j] == 0.0
                end
            end
        end
    end
    @inferred create_basis(x, degree, Val(false))
end

@testitem "create_basis_programmatic_usage" begin
    x = rand(50, 2)
    degree = 2

    # Test 5: Programmatic use with Bool
    is_ortho = true
    basis5 = create_basis(x, degree, is_ortho; center_data = false)
    @test size(basis5) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis5) == eltype(x)
    @test all(!isnan, basis5)
    @test all(!isinf, basis5)
    @inferred create_basis(x, degree, is_ortho; center_data = false)

    # Test with Bool = false
    is_ortho = false
    basis6 = create_basis(x, degree, is_ortho; center_data = true)
    @test size(basis6) == (degree + 1, degree + 1, size(x, 2))
    @inferred create_basis(x, degree, is_ortho; center_data = true)
end

@testitem "create_basis_consistency_tests" begin
    x = rand(50, 2)
    degree = 2

    # Test that default equals explicit orthonormal with centering
    basis_default = create_basis(x, degree)
    basis_explicit = create_basis(x, degree, Val(true); center_data = true)
    @test isapprox(basis_default, basis_explicit, atol = 1.0e-12)

    # Test that Bool dispatch works correctly
    basis_bool_true = create_basis(x, degree, true; center_data = true)
    basis_val_true = create_basis(x, degree, Val(true); center_data = true)
    @test isapprox(basis_bool_true, basis_val_true, atol = 1.0e-12)

    basis_bool_false = create_basis(x, degree, false)
    basis_val_false = create_basis(x, degree, Val(false))
    @test isapprox(basis_bool_false, basis_val_false, atol = 1.0e-12)
end

@testitem "create_basis_centering_parameter_tests" begin
    x = rand(50, 2)
    degree = 2

    # Test that center_data parameter actually affects orthonormal basis
    basis_centered = create_basis(x, degree, Val(true); center_data = true)
    basis_uncentered = create_basis(x, degree, Val(true); center_data = false)

    # They should be nearly identical due to backward transformation
    @test isapprox(basis_centered, basis_uncentered, atol = 1.0e-8)

    # Test that center_data is ignored for monomial basis
    basis_mono_true = create_basis(x, degree, Val(false))
    basis_mono_false = create_basis(x, degree, Val(false); center_data = false)
    @test isapprox(basis_mono_true, basis_mono_false, atol = 1.0e-12)
end

@testitem "aPCE_DerivativeBasis_recurrence_matches_forwarddiff" begin
    using ArbitraryPolynomialChaosExpansion
    using ForwardDiff
    using Random

    # Empirical-orthonormal recurrence basis from known samples (degree 8).
    rng = Xoshiro(20260806)
    x_ref = randn(rng, 10_000)
    apc = aPCE(reshape(x_ref, :, 1), 8)
    @test apc.OrthonormalBasis isa ArbitraryPolynomialChaosExpansion.RecurrenceCenteredBasis

    degs = apc.MultivariatePolynomialDegrees
    basis = apc.OrthonormalBasis
    P = apc.NumberOfTerms

    # Reference: order-fold ForwardDiff.derivative of the (univariate) Psi column.
    f_psi(ξ) = vec(aPCE_PsiPolynomialMatrix(reshape([ξ], 1, 1), degs, basis))
    ref1(x) = ForwardDiff.derivative(f_psi, x)
    ref2(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(f_psi, z), x)
    ref4(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(v -> ForwardDiff.derivative(f_psi, v), w), z), x)

    x_test = [-2.1, -0.7, 0.0, 0.4, 1.9]
    for (order, ref) in ((1, ref1), (2, ref2), (4, ref4))
        D = aPCE_DerivativeBasis(apc, x_test, order)
        @test size(D) == (P, length(x_test))
        for (j, xj) in enumerate(x_test)
            @test D[:, j] ≈ ref(xj) atol = 1e-10 rtol = 1e-10
        end
    end

    # Type stability of the recurrence path (works for runtime `order`, too).
    @test_nowarn @inferred aPCE_DerivativeBasis(apc, x_test, 2)
    @test_nowarn @inferred aPCE_DerivativeBasis(apc, x_test, 4)

    # order 0 reproduces the Psi matrix exactly (same (P, n) orientation).
    D0 = aPCE_DerivativeBasis(apc, x_test, 0)
    Ψ = aPCE_PsiPolynomialMatrix(reshape(x_test, :, 1), degs, basis)
    @test size(D0) == size(Ψ) == (P, length(x_test))
    @test D0 ≈ Ψ
end

@testitem "aPCE_DerivativeBasis_chain_rule_physical_units" begin
    using ArbitraryPolynomialChaosExpansion
    using ForwardDiff
    using Random

    # Non-unit (μ, σ) sample: 2 + 3·N(0, 1) ⇒ μ ≈ 2, σ ≈ 3. ForwardDiff
    # differentiates through the internal z = (x - μ)/σ standardization, so
    # agreement proves the (1/σ)^order chain-rule scaling is applied.
    rng = Xoshiro(7)
    x_ref = 2.0 .+ 3.0 .* randn(rng, 10_000)
    apc = aPCE(reshape(x_ref, :, 1), 6)
    @test apc.OrthonormalBasis isa ArbitraryPolynomialChaosExpansion.RecurrenceCenteredBasis
    @test apc.OrthonormalBasis.σ[1] ≈ 3.0 atol = 0.1

    degs = apc.MultivariatePolynomialDegrees
    basis = apc.OrthonormalBasis
    P = apc.NumberOfTerms
    f_psi(ξ) = vec(aPCE_PsiPolynomialMatrix(reshape([ξ], 1, 1), degs, basis))

    # Explicitly nested ForwardDiff references (orders 1-4); distinct bound
    # variables, so no closure self-reference.
    ref1(x) = ForwardDiff.derivative(f_psi, x)
    ref2(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(f_psi, z), x)
    ref3(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(f_psi, w), z), x)
    ref4(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(v -> ForwardDiff.derivative(f_psi, v), w), z), x)

    x_test = [-4.0, -1.5, 1.0, 3.7]
    for (order, ref) in ((1, ref1), (2, ref2), (3, ref3), (4, ref4))
        ref_mat = reduce(vcat, [reshape(ref(xj), 1, :) for xj in x_test])
        D = aPCE_DerivativeBasis(apc, x_test, order)
        @test D ≈ Matrix(transpose(ref_mat)) atol = 1e-9 rtol = 1e-9
    end
end

@testitem "aPCE_DerivativeBasis_monomial_backend" begin
    using ArbitraryPolynomialChaosExpansion
    using ForwardDiff
    using Random

    # Monomial / Vandermonde backend (`basis = Val(:monomial)`): x-space
    # coefficients, differentiated via derivative_coeffs + Horner.
    rng = Xoshiro(11)
    x_ref = randn(rng, 2_000)
    apc = aPCE(reshape(x_ref, :, 1), 5; basis = Val(:monomial), is_orthonormal = false)
    @test apc.OrthonormalBasis isa Array{Float64, 3}

    degs = apc.MultivariatePolynomialDegrees
    basis = apc.OrthonormalBasis
    P = apc.NumberOfTerms
    f_psi(ξ) = vec(aPCE_PsiPolynomialMatrix(reshape([ξ], 1, 1), degs, basis))

    ref1(x) = ForwardDiff.derivative(f_psi, x)
    ref2(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(f_psi, z), x)
    ref3(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(f_psi, w), z), x)
    ref4(x) = ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> ForwardDiff.derivative(v -> ForwardDiff.derivative(f_psi, v), w), z), x)

    x_test = [-1.4, -0.2, 0.6, 1.1]
    for (order, ref) in ((1, ref1), (2, ref2), (3, ref3), (4, ref4))
        D = aPCE_DerivativeBasis(apc, x_test, order)
        @test size(D) == (P, length(x_test))
        for (j, xj) in enumerate(x_test)
            @test D[:, j] ≈ ref(xj) atol = 1e-10 rtol = 1e-10
        end
    end

    # order 0 equals the Psi matrix on the monomial backend, too.
    D0 = aPCE_DerivativeBasis(apc, x_test, 0)
    Ψ = aPCE_PsiPolynomialMatrix(reshape(x_test, :, 1), degs, basis)
    @test D0 ≈ Ψ
end
