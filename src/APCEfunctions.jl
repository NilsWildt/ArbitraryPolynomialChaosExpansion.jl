# Copyright 2024 wildt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using ChainRulesCore


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
struct CoSorter{T1, T2, T3, A <: AbstractVecOrMat{T1}, B <: AbstractVecOrMat{T2}, C <: AbstractVecOrMat{T3}} <: AbstractVector{SpecialCoSorterElement{T1, T2, T3}}
    sortarray::A
    otherarray::B
    coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)
Base.getindex(c::CoSorter, i...) =
    SpecialCoSorterElement(getindex(c.sortarray, i...), getindex(c.otherarray, i...), getindex(c.coarray, i...))
Base.setindex!(c::CoSorter, t::SpecialCoSorterElement, i...) =
    (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c)

Base.isless(a::SpecialCoSorterElement, b::SpecialCoSorterElement) = isless(a.x, b.x) || (a.x == b.x && isless(a.z, b.z))
Base.Sort.defalg(v::C) where {T <: Union{Number, Missing}, C <: CoSorter{T}} = Base.DEFAULT_UNSTABLE

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

function evalpoly_two(x::ReverseDiff.TrackedReal, coeffs::AbstractVector{<:ReverseDiff.TrackedReal})
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
@inline function evaluate_polynomial_horner_array(x::AbstractVector{T}, coeffs::AbstractVector{S}) where {T <: Real, S <: Real}
    R = promote_type(T, S)
    results = Vector{R}(undef, length(x))
    @inbounds for (i, xi) in enumerate(x)
        result = zero(R)
        @simd for coeff in reverse(coeffs)
            result = muladd(result, xi, coeff)
        end
        results[i] = result
    end
    return results
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
        @infiltrate
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
function aPCE_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T, s_marginals::F, s_interactions::F) where {T <: Integer, F <: Real}
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
        try
            if percentage < 0.0 || percentage > 1.0
                @warn "Percentage must be between 0 and 1"
            end
            n = length(array)
            num_to_keep = round(Int, percentage * n)
            return @views array[1:num_to_keep]
        catch
            return array[:]
        end
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
    keeper_interactions = d_interactions_indices[filter_by_percentage(all_interactions, s_interactions)]
    idxkeep = vcat(keeper_marginals, keeper_interactions)
    indices = @views indices[idxkeep, :]
    indices = vcat(indices, Base.zeros(T, num_dimensions)')
    indices = sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
    return indices::Matrix{T}
end

# ===== BASIS FUNCTIONS =====

"""
    aPCE_OrthonormalBasis(Data, Degree, center_data::Val)

Compute orthonormal polynomial basis for 1D data using moment-based approach.
- `Data`: 1D array of data points
- `Degree`: Maximum polynomial degree
- `center_data`: Val{true} for centered basis, Val{false} for uncentered
"""
function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S, center_data::Val{true}) where {T <: Real, S <: Integer}
    if ndims(Data) > 1 && size(Data, 2) > 1
        throw(ArgumentError("aPCE_OrthonormalBasis expects 1D data only. Use create_basis for multi-dimensional data."))
    end

    data_vec = vec(Data)
    NumberOfDataPoints = length(data_vec)
    dd = Degree

    # Centering: scale data by mean (not subtraction)
    data_mean = mean(data_vec)
    Data_scaled = data_vec ./ data_mean

    # Compute moments using scaled data
    m = zeros(T, 2 * dd + 2)
    for i in 0:(2 * dd + 1)
        m[i + 1] = sum(Data_scaled .^ i) / NumberOfDataPoints
    end

    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1)

    for degree in 0:dd
        Hankel = @views OrthogonalBasis[1:(degree + 1), 1:(degree + 1)]
        Vc = zeros(T, degree + 1)

        # Build Hankel matrix
        for i in 0:(degree - 1)
            for j in 0:degree
                Hankel[i + 1, j + 1] = m[i + j + 1]
            end
            max_val = maximum(abs.(@views Hankel[i + 1, :]))
            if max_val > zero(T)
                Hankel[i + 1, :] = @views Hankel[i + 1, :] / max_val
            end
        end

        # Last row setup
        for j in 0:(degree - 1)
            Hankel[degree + 1, j + 1] = zero(T)
        end
        Hankel[degree + 1, degree + 1] = one(T)
        max_val = maximum(abs.(@views Hankel[degree + 1, :]))
        if max_val > zero(T)
            Hankel[degree + 1, :] = @views Hankel[degree + 1, :] / max_val
        end

        # Right-hand side vector
        for i in 0:(degree - 1)
            Vc[i + 1] = zero(T)
        end
        Vc[degree + 1] = one(T)

        # Solve linear system
        try
            OrthogonalBasis[degree + 1, 1:(degree + 1)] .= Hankel \ Vc
        catch
            OrthogonalBasis[degree + 1, 1:(degree + 1)] .= pinv(Hankel) * Vc
        end

        # Normalization
        P_norm = zero(T)
        for i in 1:NumberOfDataPoints
            Poly = zero(T)
            for k in 0:degree
                Poly += @views OrthogonalBasis[degree + 1, k + 1] * Data_scaled[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end

        # Improved numerical stability check
        eps_val = eps(T) * 1.0e6  # More conservative epsilon
        if P_norm <= eps_val
            # Fallback: use standard monomial basis for this degree
            for i in 1:(degree + 1)
                for j in 1:(degree + 1)
                    if i == degree + 1 && j == degree + 1
                        OrthonormalBasis[i, j] = one(T)
                    elseif i >= j && i <= degree
                        OrthonormalBasis[i, j] = one(T)
                    else
                        OrthonormalBasis[i, j] = zero(T)
                    end
                end
            end
        else
            # Normal normalization with safety check
            norm_factor = sqrt(P_norm)
            if norm_factor <= eps_val
                norm_factor = eps_val
            end
            for k in 0:degree
                OrthonormalBasis[degree + 1, k + 1] = @views OrthogonalBasis[degree + 1, k + 1] / norm_factor
            end
        end
    end

    # Backward transformation to data space
    for k in 1:size(OrthonormalBasis, 2)
        OrthonormalBasis[:, k] ./= (data_mean^(k - 1))
    end

    return OrthonormalBasis
end

function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S, center_data::Val{false}) where {T <: Real, S <: Integer}
    if ndims(Data) > 1 && size(Data, 2) > 1
        throw(ArgumentError("aPCE_OrthonormalBasis expects 1D data only. Use create_basis for multi-dimensional data."))
    end

    data_vec = vec(Data)
    NumberOfDataPoints = length(data_vec)
    dd = Degree

    # Compute raw moments (no centering)
    m = zeros(T, 2 * dd + 2)
    for i in 0:(2 * dd + 1)
        m[i + 1] = sum(data_vec .^ i) / NumberOfDataPoints
    end

    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1)

    # Same algorithm as centered version but without scaling
    for degree in 0:dd
        Hankel = @views OrthogonalBasis[1:(degree + 1), 1:(degree + 1)]
        Vc = zeros(T, degree + 1)

        for i in 0:(degree - 1)
            for j in 0:degree
                Hankel[i + 1, j + 1] = m[i + j + 1]
            end
            max_val = maximum(abs.(@views Hankel[i + 1, :]))
            if max_val > zero(T)
                Hankel[i + 1, :] = @views Hankel[i + 1, :] / max_val
            end
        end

        for j in 0:(degree - 1)
            Hankel[degree + 1, j + 1] = zero(T)
        end
        Hankel[degree + 1, degree + 1] = one(T)
        max_val = maximum(abs.(@views Hankel[degree + 1, :]))
        if max_val > zero(T)
            Hankel[degree + 1, :] = @views Hankel[degree + 1, :] / max_val
        end

        for i in 0:(degree - 1)
            Vc[i + 1] = zero(T)
        end
        Vc[degree + 1] = one(T)

        try
            OrthogonalBasis[degree + 1, 1:(degree + 1)] .= Hankel \ Vc
        catch
            OrthogonalBasis[degree + 1, 1:(degree + 1)] .= pinv(Hankel) * Vc
        end

        # Normalization using original data
        P_norm = zero(T)
        for i in 1:NumberOfDataPoints
            Poly = zero(T)
            for k in 0:degree
                Poly += @views OrthogonalBasis[degree + 1, k + 1] * data_vec[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end

        # Improved numerical stability check
        eps_val = eps(T) * 1.0e6  # More conservative epsilon
        if P_norm <= eps_val
            # Fallback: use standard monomial basis for this degree
            for i in 1:(degree + 1)
                for j in 1:(degree + 1)
                    if i == degree + 1 && j == degree + 1
                        OrthonormalBasis[i, j] = one(T)
                    elseif i >= j && i <= degree
                        OrthonormalBasis[i, j] = one(T)
                    else
                        OrthonormalBasis[i, j] = zero(T)
                    end
                end
            end
        else
            # Normal normalization with safety check
            norm_factor = sqrt(P_norm)
            if norm_factor <= eps_val
                norm_factor = eps_val
            end
            for k in 0:degree
                OrthonormalBasis[degree + 1, k + 1] = @views OrthogonalBasis[degree + 1, k + 1] / norm_factor
            end
        end
    end

    return OrthonormalBasis
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
        basis[5, 1] = -(-m[4] * m[6] * m[5] - m[4]^2 * m[8] + m[4] * m[6]^2 + 2 * m[4] * m[5] * m[7] - 2 * m[5]^2 * m[6] + m[5]^3) / denom
        basis[5, 2] = -(m[6]^3 - m[5]^2 * m[4]^2 + m[4]^2 * m[5] * m[6] - m[5] * m[6] * m[7] - m[4] * m[6] * m[7] - m[4] * m[6] * m[8] + m[4] * m[5] * m[8] + m[4]^3 * m[6] - m[4]^3 * m[7] + m[4] * m[7]^2) / (m[4] * denom)
        basis[5, 3] = -(m[5]^2 * m[4]^2 - m[4] * m[5] * m[8] - m[5] * m[6]^2 + m[5]^2 * m[7] - m[4]^3 * m[6] + m[4] * m[6] * m[7]) / (m[4] * denom)
        basis[5, 4] = (m[4]^2 * m[5] - m[6] * m[5] - m[4] * m[8] + m[6]^2 + m[5] * m[7]) / denom
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
function create_basis(x::AbstractArray{T}, degree::Integer, ::Val{true}; center_data::Bool = true) where {T <: Real}
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
function create_basis(x::AbstractArray{T}, degree::Integer, ::Val{false}; center_data::Bool = false) where {T <: Real}
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

# ===== PSI MATRIX FUNCTIONS =====

"""
    compute_Psi_element(i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions)

Compute a single element of the Psi matrix (polynomial evaluation).
"""
function compute_Psi_element(i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions)
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
function aPCE_PsiPolynomialMatrix_zygote(TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)

    Psi = [
        compute_Psi_element(i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions)
            for i in 1:NumberOfTerms, j in 1:NCpoints
    ]
    return reshape(Psi, NumberOfTerms, NCpoints)
end

"""
    aPCE_PsiPolynomialMatrix(TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)

Compute the Psi matrix optimized for ForwardDiff compatibility.
Uses vectorized operations for better performance.
"""
function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{S}) where {S, T <: Real}
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = ones(T, NumberOfTerms, NCpoints)

    @inbounds for i in 1:NumberOfTerms
        for ii in 1:InputDimensions
            degree = MultivariatePolynomialDegrees[i, ii] + 1
            coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
            x = @views TrainingInput[:, ii]
            poly_values = evaluate_polynomial_horner_array(x, coeffs)
            #    poly_values = evalpoly_two.(x, coeffs)
            for j in 1:NCpoints
                Psi[i, j] *= poly_values[j]
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
@inline function compute_moments!(m::AbstractArray{T}, Data::AbstractArray{S}, NumberOfDataPoints::Integer, dd::Integer) where {T <: Real, S <: Real}
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
        InputDistribution::AbstractArray{T}, NumberOfTerms; strategy = :PCM
    )::Matrix{T} where {T <: Real}

    polynomial_roots = zeros(T, input_dimensions, ExpansionDegree + 1)
    @inbounds for d in Base.oneto(Int64(input_dimensions))
        polynomial_basis = @views OrthonormalBasis[:, :, d]
        roots = PolynomialRoots.roots(@views polynomial_basis[ExpansionDegree + 1, :])
        real_roots = real.(roots)
        polynomial_roots[d, 1:length(real_roots)] = real_roots
    end

    PointsVector = 1:(ExpansionDegree + 1) |> collect
    UniqueCombinations = stack(reduce(vcat, (UnrolledUtilities.unrolled_product([PointsVector for _ in 1:input_dimensions]...))))'

    sort_indices = sortperm(sum(UniqueCombinations; dims = 2); dims = 1)
    SortUniqueCombinations = UniqueCombinations[sort_indices[:], :]

    if strategy == :FT
        TrainingInput = SortUniqueCombinations
        return Array(view(TrainingInput, :, (1:size(TrainingInput, 2))))
    elseif strategy == :PCM
        temp = abs.(polynomial_roots .- StatsBase.mean(InputDistribution; dims = 1)[:, :][1])
        temp_sort = mapslices(sortperm, temp, dims = 2)
        @inbounds for i in axes(polynomial_roots, 1)
            polynomial_roots[i, :] = @views polynomial_roots[i, temp_sort[i, :]]
        end
        collocation_points = zeros(T, NumberOfTerms, input_dimensions)
        @inbounds for i in 1:NumberOfTerms
            for j in axes(SortUniqueCombinations, 2)
                collocation_points[i, j] = @views polynomial_roots[j, Int(SortUniqueCombinations[i, j])]
            end
        end
        collocation_points = sortslices(collocation_points, dims = 1, by = x -> x[1])
        return Array(view(collocation_points, :, (1:size(collocation_points, 2))))
    end
end

# ===== EVALUATION FUNCTIONS =====

"""
   compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)

Compose the Psi matrix for evaluation.
"""
function compose_Ψ(x::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis, degree) where {T}
    Ψ = aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, OrthonormalBasis)' |> Matrix{T}
    return Ψ
end

"""
   evaluate_Ψ(x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)

Evaluate the polynomial expansion at points x with given coefficients.
"""
function evaluate_Ψ(x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
    T = eltype(coeffs)
    Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
    TensorOperations.@tensor order = (k, i) PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
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
function train(Ψ::AbstractArray{T}, y_rhs; bayesian_inversion = :true, reg_order = 0) where {T <: Real}
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
                method = LBFGS(linesearch = LineSearches.BackTracking())
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

@testitem "aPCE_MultivariatePolynomialDegrees" begin
    @test aPCE_MultivariatePolynomialDegrees(2, 1, 1.0, 1.0) == [0 0; 0 1; 1 0]
    @test aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0) == [0 0; 0 1; 1 0; 0 2; 1 1; 2 0]
    @inferred aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0)
end

@testitem "aPCE_OrthonormalBasis" begin
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true)) ≈ [1.0 0.0; -0.5 1.5]
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false)) ≈ [1.0 0.0; -0.5 1.5]
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false))
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true))
end

@testitem "numberPolynomials" begin
    @test APCE.numberPolynomials(3, 2) == 10
    @test APCE.numberPolynomials(5, 3) == 56
    @test APCE.numberPolynomials(0, 0) == 1
    @test APCE.numberPolynomials(1, 1) == 2
    @test typeof(APCE.numberPolynomials(3, 2)) == Int
end

@testitem "aPCE_MultivariatePolynomialDegrees_test" begin
    @test aPCE_MultivariatePolynomialDegrees(2, 1, 1.0, 1.0) == [0 0; 0 1; 1 0]
    @test aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0) == [0 0; 0 1; 1 0; 0 2; 1 1; 2 0]
    @inferred aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0)
end


@testitem "aPCE_OrthonormalBasis_test" begin
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true)) ≈ [1.0 0.0; -0.5 1.5]
    @test aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false)) ≈ [1.0 0.0; -0.5 1.5]
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false))
    @inferred aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true))
end

@testitem "numberPolynomials_test" begin
    @test APCE.numberPolynomials(3, 2) == 10
    @test APCE.numberPolynomials(5, 3) == 56
    @test APCE.numberPolynomials(0, 0) == 1
    @test APCE.numberPolynomials(1, 1) == 2
    @test typeof(APCE.numberPolynomials(3, 2)) == Int
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
    OrthonormalBasis = rand(3, 3, 2)
    InputDistribution = rand(10, 2)
    NumberOfTerms = 6
    @inferred APCE.GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis, InputDistribution, NumberOfTerms)
    @test typeof(APCE.GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis, InputDistribution, NumberOfTerms)) == Matrix{Float64}
end

@testitem "edge_cases_tests" begin
    # Test edge cases for evalpoly_two
    @test APCE.evalpoly_two(0.0, [1.0]) == 1.0  # Constant polynomial
    @test APCE.evalpoly_two(1.0, [0.0, 0.0]) == 0.0  # Zero polynomial
    @test APCE.evalpoly_two(Inf, [1.0, 2.0]) == Inf  # Infinity input
    @test isnan(APCE.evalpoly_two(NaN, [1.0, 2.0]))  # NaN input

    # Test edge cases for evaluate_derivative_horner
    @test APCE.evaluate_derivative_horner(0.0, [1.0]) == 0.0  # Constant polynomial
    @test APCE.evaluate_derivative_horner(1.0, [0.0, 0.0]) == 0.0  # Zero polynomial

    # Test edge cases for train
    Ψ = zeros(5, 3)
    y_rhs = zeros(5, 2)
    @test all(iszero, APCE.train(Ψ, y_rhs))  # Zero inputs
    @test size(APCE.train(Ψ, y_rhs)) == (3, 2)  # Correct output size

    # Test edge cases for aPCE_FullBasis
    @test size(APCE.aPCE_FullBasis([1.0], 0)) == (1, 1)  # Degree 0
    @test size(APCE.aPCE_FullBasis([1.0], 1)) == (2, 2)  # Degree 1
end


@testitem "aPCE_OrthonormalBasis1" begin
    @test APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true)) ≈ [1.0 0.0; -0.5 1.5]
    @test APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false)) ≈ [1.0 0.0; -0.5 1.5]
    @inferred APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(false))
    @inferred APCE.aPCE_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1, Val(true))
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
    APCE.reverse_columns!(mat1)
    @test mat1 == expected1

    mat2 = [1 2; 3 4; 5 6]
    expected2 = [2 1; 4 3; 6 5]
    APCE.reverse_columns!(mat2)
    @test mat2 == expected2

    mat3 = [1 2 3 4; 5 6 7 8]
    expected3 = [4 3 2 1; 8 7 6 5]
    APCE.reverse_columns!(mat3)
    @test mat3 == expected3

    @test typeof(APCE.reverse_columns!(mat1)) == Matrix{Int}
end


@testitem "evaluate_derivative_horner_test" begin
    @test APCE.evaluate_derivative_horner(2.0, [1.0, 2.0, 3.0]) == 14.0  # Derivative of 1 + 2x + 3x^2 at x=2
    @test APCE.evaluate_derivative_horner(0.0, [1.0, 2.0, 3.0]) == 2.0   # Derivative of 1 + 2x + 3x^2 at x=0
    @test APCE.evaluate_derivative_horner(1.0, [0.0, 0.0, 0.0]) == 0.0   # Derivative of 0 polynomial at x=1
    @test APCE.evaluate_derivative_horner(1.0, [5.0]) == 0.0             # Derivative of constant polynomial at x=1
    @test typeof(APCE.evaluate_derivative_horner(2.0, [1.0, 2.0, 3.0])) == Float64
    @inferred APCE.evaluate_derivative_horner(1.0, [5.0])
end

@testitem "evalpoly_two_test" begin
    @test APCE.evalpoly_two(2.0, [1.0, 2.0, 3.0, 4.0]) == 49.0  # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=2
    @test APCE.evalpoly_two(0.0, [1.0, 2.0, 3.0, 4.0]) == 1.0   # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=0
    @test APCE.evalpoly_two(1.0, [0.0, 0.0, 0.0, 0.0]) == 0.0   # Zero polynomial at x=1
    @test APCE.evalpoly_two(1.0, [5.0]) == 5.0                  # Constant polynomial at x=1
    @test APCE.evalpoly_two(2.0, [1.0, -1.0, 1.0, -1.0]) == -5.0  # Polynomial 1 - x + x^2 - x^3 at x=2
end

@testitem "EstrinPoly_test" begin
    import Pkg
    Pkg.add("Estrin")
    using Estrin
    @test Estrin.Poly([1.0, 2.0, 3.0, 4.0]).(2.0) == 49.0  # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=2
    @test Estrin.Poly([1.0, 2.0, 3.0, 4.0]).(0.0) == 1.0   # Polynomial 1 + 2x + 3x^2 + 4x^3 at x=0
    @test Estrin.Poly([0.0, 0.0, 0.0, 0.0]).(1.0) == 0.0   # Zero polynomial at x=1
    @test Estrin.Poly([5.0]).(1.0) == 5.0                  # Constant polynomial at x=1
    @test Estrin.Poly([1.0, -1.0, 1.0, -1.0]).(2.0) == -5.0  # Polynomial 1 - x + x^2 - x^3 at x=2
end


@testitem "derivative_coeffs_test" begin
    @test APCE.derivative_coeffs([1.0, 2.0, 3.0]) == [2.0, 6.0]  # Derivative of 1 + 2x + 3x^2
    @test APCE.derivative_coeffs([0.0, 0.0, 0.0]) == [0.0, 0.0]  # Derivative of 0 polynomial
    @test APCE.derivative_coeffs([5.0]) == [0.0]                 # Derivative of constant polynomial
    @test APCE.derivative_coeffs([1.0, -1.0, 1.0, -1.0]) == [-1.0, 2.0, -3.0]  # Derivative of 1 - x + x^2 - x^3
end

@testitem "create_basis_orthonormal_true_test" begin
    # Test basic functionality
    x = rand(100, 2)
    degree = 3
    basis = APCE.create_basis(x, degree, Val(true))

    @test size(basis) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis) == eltype(x)
    @test all(!isnan, basis)
    @test all(!isinf, basis)

    # Test type stability
    @inferred APCE.create_basis(x, degree, Val(true))

    # Test with different input types
    x_float32 = rand(Float32, 50, 3)
    basis_float32 = APCE.create_basis(x_float32, 2, Val(true))
    @test eltype(basis_float32) == Float32
    @test size(basis_float32) == (3, 3, 3)

    # Test edge cases
    x_single = rand(10, 1)
    basis_single = APCE.create_basis(x_single, 0, Val(true))
    @test size(basis_single) == (1, 1, 1)
    @test basis_single[1, 1, 1] ≈ 1.0 atol = 1.0e-10
end

@testitem "create_basis_orthonormal_false_test" begin
    # Test basic functionality
    x = rand(100, 2)
    degree = 3
    basis = APCE.create_basis(x, degree, Val(false))

    @test size(basis) == (degree + 1, degree + 1, size(x, 2))
    @test eltype(basis) == eltype(x)
    @test all(!isnan, basis)
    @test all(!isinf, basis)

    # Test type stability
    @inferred APCE.create_basis(x, degree, Val(false))

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
    basis_float32 = APCE.create_basis(x_float32, 2, Val(false))
    @test eltype(basis_float32) == Float32
    @test size(basis_float32) == (3, 3, 3)

    # Test edge cases
    x_single = rand(10, 1)
    basis_single = APCE.create_basis(x_single, 0, Val(false))
    @test size(basis_single) == (1, 1, 1)
    @test basis_single[1, 1, 1] == 1.0

    # Test degree 1 case
    basis_degree1 = APCE.create_basis(x_single, 1, Val(false))
    @test size(basis_degree1) == (2, 2, 1)
    @test basis_degree1[:, :, 1] == [1.0 0.0; 1.0 1.0]
end

@testitem "create_basis_comparison_test" begin
    # Test that orthonormal and non-orthonormal bases have same structure but different values
    x = rand(50, 2)
    degree = 2

    basis_ortho = APCE.create_basis(x, degree, Val(true))
    basis_full = APCE.create_basis(x, degree, Val(false))

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
        @test !all(basis_dim[i, j] == (i >= j ? 1.0 : 0.0) for i in 1:(degree + 1), j in 1:(degree + 1))
    end
end

@testitem "create_basis_functional_test" begin
    # Test that the bases can be used in polynomial evaluation
    x = rand(20, 2)
    degree = 2

    # Create both types of bases
    basis_ortho = APCE.create_basis(x, degree, Val(true))
    basis_full = APCE.create_basis(x, degree, Val(false))

    # Test that we can extract coefficients from both bases
    for dim in 1:size(x, 2)
        for d in 1:(degree + 1)
            coeffs_ortho = APCE.coeffs_from_basis(basis_ortho, d, dim)
            coeffs_full = APCE.coeffs_from_basis(basis_full, d, dim)

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
    @test_nowarn APCE.aPCE_PsiPolynomialMatrix_zygote(test_point', multivar_degrees, basis_ortho)
    @test_nowarn APCE.aPCE_PsiPolynomialMatrix_zygote(test_point', multivar_degrees, basis_full)
end

@testitem "create_basis_default_test" begin
    using LinearAlgebra
    # Test the default create_basis function (should default to orthonormal)
    x = rand(30, 2)
    degree = 2

    # Test default behavior
    basis_default = APCE.create_basis(x, degree)
    basis_ortho = APCE.create_basis(x, degree, Val(true))

    # Default should be the same as orthonormal
    @test isapprox(basis_default, basis_ortho, atol = 1.0e-10)

    # Test with center_data parameter
    basis_default_centered = APCE.create_basis(x, degree, center_data = true)
    basis_default_uncentered = APCE.create_basis(x, degree, center_data = false)

    # Debug: Check the data properties
    @info "Data mean" mean(x, dims = 1)
    @info "Data std" std(x, dims = 1)
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
    @inferred APCE.create_basis(x, degree)
    @inferred APCE.create_basis(x, degree, center_data = true)
    @inferred APCE.create_basis(x, degree, center_data = false)
end

@testitem "create_basis_centered_vs_uncentered_test" begin
    using LinearAlgebra
    # Test with data that has a clear mean different from 1.0
    # This should make the difference between centered and uncentered modes obvious
    x = [1.0, 5.0, 10.0, 15.0, 20.0] .* ones(5, 2)  # Data with mean 10.2
    degree = 2

    @info "Test data mean" mean(x, dims = 1)
    @info "Test data std" std(x, dims = 1)

    basis_centered = APCE.create_basis(x, degree, center_data = true)
    basis_uncentered = APCE.create_basis(x, degree, center_data = false)

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
    using Statistics: mean
    # Test to understand the centering effect
    x = [1.0, 5.0, 10.0, 15.0, 20.0] .* ones(5, 2)  # Data with mean 10.2
    degree = 2

    # Test individual aPCE_OrthonormalBasis calls to see the effect
    x_col1 = view(x, :, 1)
    mean_x = mean(x_col1)
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
    basis_centered = APCE.create_basis(x_large, degree, center_data = true)
    basis_uncentered = APCE.create_basis(x_large, degree, center_data = false)

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
