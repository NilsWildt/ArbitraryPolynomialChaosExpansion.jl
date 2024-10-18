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

@info "Benchmarking Matrix mutplication speed" LinearAlgebra.peakflops(; parallel = true)

# Strided.set_num_threads(CPUSummary.get_cpu_threads() ÷ 2)
LinearAlgebra.BLAS.set_num_threads(CPUSummary.get_cpu_threads() ÷ 2)

@stable function normalization_functions(matrix)
    # Calculate mean and std for each column
    col_means = StatsBase.mean(matrix, dims = 1)
    col_stds = StatsBase.std(matrix, dims = 1)
    # Define the normalization function
    normalize = (x) -> (x .- col_means) ./ col_stds
    # Define the inverse normalization function
    inverse_normalize = (x) -> x .* col_stds .+ col_means
    return normalize, inverse_normalize
end

# Sort mean and variance at same time
struct SpecialCoSorterElement{T1, T2, T3}
    x::T1
    z::T2
    y::T3
end

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


Base.Sort.defalg(v::C) where {T <: Union{Number, Missing}, C <: CoSorter{T}} =
    Base.DEFAULT_UNSTABLE
@stable function special_sort_two_arrays!(x::AbstractArray, y::AbstractArray)
    T = CoSorter(x[:, 1], x[:, 2], y)
    sort!(T)
    x = T.sortarray
    y = T.coarray
end


@stable function aPCE_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T, s_marginals::F, s_interactions::F) where {T <: Integer, F <: Real}
    # Initialize the indices for the first parameter
    @stable function get_stats(r)::Array{F} # Returns sum, nzeros, mean, var, min,max
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
        # Ensure the percentage is within the valid range
        try
            if percentage < 0.0 || percentage > 1.0
                # throw(ArgumentError("Percentage must be between 0 and 1"))
                @warn "Percentage must be between 0 and 1"
            end
            n = length(array)
            num_to_keep = round(Int, percentage * n)
            return @views array[1:num_to_keep]
        catch
            return  array[:]
        end
    end


    range_ = 0:max_degree  # |> sparse
    indices = reshape(range_, :, 1)
    for di in 1:(num_dimensions - 1)
        indices = repeat(indices, inner = (max_degree + 1, 1))
        front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di) |> sparse
        indices = ApplyArray(hcat, front, indices)
        indices = @~ indices[vec(sum(indices; dims = 2)) .<= max_degree, :]
    end

    stats = reduce(hcat, map(x -> get_stats(x), eachrow(indices)))'
    d_marginal_indices = T[]
    d_interactions_indices = T[]
    @inbounds for r in axes(stats, 1) # Go over columns
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
    # reverse_columns!(indices)
    return indices::Matrix{T} # from sparse to matrix.
end


@stable function aPCE_PsiPolynomialMatrix_zygote(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis) where {T <: Real}
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = Zygote.bufferfrom(ones(eltype(TrainingInput), NumberOfTerms, NCpoints))
    # OrthonormalBasis = T.(OrthonormalBasis)
    # Function to evaluate polynomials for a given term and input sample
    @inbounds for i in 1:NumberOfTerms  # For each term in the polynomial expansion
        # product = 1.0  # Initialize the product for this term and sample
        for ii in 1:InputDimensions  # For each dimension of the input
            degree = MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
            coeffs = @views OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
            # p = Polynomials.Polynomial{T}(coeffs)  # Create the polynomial
            # p = Poly(coeffs)
            for j in 1:NCpoints  # For each input sample
                x = TrainingInput[j, ii]
                Psi[i, j] *= evalpoly_two(x, coeffs)
                # Psi[i,j] *= evalpoly(x, p)
            end
        end
    end
    return copy(Psi)
end

# Need this function for Orthonormal Basis is a ForwardDiff.
@stable function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{S}) where {S, T <: Real}
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = ones(T, NumberOfTerms, NCpoints)
    # OrthonormalBasis = T.(OrthonormalBasis)
    # Function to evaluate polynomials for a given term and input sample
    @inbounds for i in 1:NumberOfTerms  # For each term in the polynomial expansion
        # product = 1.0  # Initialize the product for this term and sample
        for ii in 1:InputDimensions  # For each dimension of the input
            degree = MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
            coeffs = @views OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
            p = Polynomials.Polynomial(coeffs)  # Create the polynomial
            # p = Poly(coeffs)
            x = @views TrainingInput[:, ii]
            @.. Psi[i, :] *= p(x)  # Evaluate the polynomial at x and multiply
            # Psi[i,j] *= evalpoly(x, p)
        end
    end
    return Psi
end

@stable @inline function compute_moments!(m::AbstractArray{T}, Data::AbstractArray{T}, NumberOfDataPoints, dd) where {T <: Real}
    current_power = ones(T, length(Data))  # Start with Data .^ 0 which is 1
    @inbounds for l in 0:(2 * dd + 1)
        m[l + 1] = sum(current_power) / NumberOfDataPoints
        current_power .*= Data  # Increment the power of Data
    end
end


@stable @inbounds function aPCE_OrthonormalBasis(Data, Degree::S, normalize_data::Val{true}) where {S <: Integer}
    T = eltype(Data)
    d = Degree #Degree of polynomial expansion
    dd = d #Degree of polinomial for roots definitions
    # @warn "Weird, is that properly normalized? Not /std and minus mean?"
    NumberOfDataPoints = length(Data)
    MeanOfData = mean(Data)
    Data = Data ./ MeanOfData
    m = zeros(T, 2 * dd + 2)
    @batch for col in axes(Data, 2)
        compute_moments!(m, view(Data, :, col), NumberOfDataPoints, dd)
    end
    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1) # Allocate once for all :)
    PolyCoeff_NonNorm_prealloc = zeros(T, dd + 1, dd + 1) # Allocate once for all :)

    for degree in 0:dd
        Hankel = @views OrthogonalBasis[1:(degree + 1), 1:(degree + 1)]
        Vc = zeros(T, degree + 1)
        PolyCoeff_NonNorm = @views PolyCoeff_NonNorm_prealloc[1:(degree + 1), 1:(degree + 1)]

        for i in 0:(degree - 1)
            for j in 0:degree
                Hankel[i + 1, j + 1] = @views m[i + j + 1]  # put in the moment
            end
            Hankel[i + 1, :] = @views Hankel[i + 1, :] / maximum(abs.(@views Hankel[i + 1, :]))
        end
        for j in 0:(degree - 1)
            Hankel[degree + 1, j + 1] = zero(T)
        end
        Hankel[degree + 1, degree + 1] = one(T)
        Hankel[degree + 1, :] = @views Hankel[degree + 1, :] / maximum(abs.(@views Hankel[degree + 1, :]))

        # Loop for Vc
        for i in 0:(degree - 1)
            Vc[i + 1] = zero(T)
        end
        Vc[degree + 1] = one(T)


        # PolyCoeff_NonNorm[degree+1, 1:degree+1] .= LinearAlgebra.factorize(Hankel) \ Vc

        # PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Krylov.usymlq(Hankel,Vc) |> first
        # catch
        #     @warn "Hankel matrix singular, trying pseudo inverse." #  Vp Hankel Vc
        PolyCoeff_NonNorm[degree + 1, 1:(degree + 1)] .= pinv(Hankel) * Vc # more robust?
        # end
        # Vp = Hankel \ Vc
        # PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Vp
        # @ignore_derivatives begin

        # deviation = 100 * abs(sum(abs.(Hankel * PolyCoeff_NonNorm[degree+1, 1:degree+1])) - sum(abs.(Vc)))
        # if (deviation > 0.5)
        #     @warn "Computational error of the linear solver is too high: $(round(deviation;digits=3))"
        # end

        #Normalization of polynomial coefficients
        P_norm = 0.0
        for i in 1:NumberOfDataPoints
            Poly = 0
            for k in 0:degree
                @fastmath Poly += @views PolyCoeff_NonNorm[degree + 1, k + 1] * Data[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end
        for k in 0:degree
            @fastmath OrthonormalBasis[degree + 1, k + 1] = @views PolyCoeff_NonNorm[degree + 1, k + 1] / sqrt(P_norm)
        end
    end
    for k in 1:lastindex(OrthonormalBasis, 2)
        @fastmath OrthonormalBasis[:, k] = @views OrthonormalBasis[:, k] ./ (MeanOfData^(k - 1))
    end
    return OrthonormalBasis
end


@stable @inbounds function aPCE_OrthonormalBasis(Data, Degree::S, normalize_data::Val{false}) where {S <: Integer}
    T = eltype(Data)
    d = Degree #Degree of polinomial expansion
    dd = d #Degree of polynomial for roots definition
    NumberOfDataPoints = length(Data)

    m = zeros(T, 2 * dd + 2)
    @batch for col in axes(Data, 2)
        compute_moments!(m, view(Data, :, col), NumberOfDataPoints, dd)
    end
    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1) # Allocate once for all :)
    PolyCoeff_NonNorm_prealloc = zeros(T, dd + 1, dd + 1) # Allocate once for all :)

    for degree in 0:dd
        Hankel = @views OrthogonalBasis[1:(degree + 1), 1:(degree + 1)]
        Vc = zeros(T, degree + 1)
        PolyCoeff_NonNorm = @views PolyCoeff_NonNorm_prealloc[1:(degree + 1), 1:(degree + 1)]

        for i in 0:(degree - 1)
            for j in 0:degree
                Hankel[i + 1, j + 1] = @views m[i + j + 1]  # put in the moment
            end
            Hankel[i + 1, :] = @views Hankel[i + 1, :] / maximum(abs.(@views Hankel[i + 1, :]))
        end
        for j in 0:(degree - 1)
            Hankel[degree + 1, j + 1] = zero(T)
        end
        Hankel[degree + 1, degree + 1] = one(T)
        Hankel[degree + 1, :] = @views Hankel[degree + 1, :] / maximum(abs.(@views Hankel[degree + 1, :]))

        # Loop for Vc
        for i in 0:(degree - 1)
            Vc[i + 1] = zero(T)
        end
        Vc[degree + 1] = one(T)

        PolyCoeff_NonNorm[degree + 1, 1:(degree + 1)] .= Hankel \ Vc #  pinv(Hankel) * Vc #

        P_norm = 0.0
        for i in 1:NumberOfDataPoints
            Poly = 0
            for k in 0:degree
                Poly += @views PolyCoeff_NonNorm[degree + 1, k + 1] * Data[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end
        for k in 0:degree
            OrthonormalBasis[degree + 1, k + 1] = @views PolyCoeff_NonNorm[degree + 1, k + 1] / sqrt(P_norm)
        end
    end

    return OrthonormalBasis
end


@stable @inbounds function aPCE_OrthonormalBasis_zygote(Data, Degree::S, normalize_data::Val{false}) where {S <: Integer}
    T = eltype(Data)
    d = Degree #Degree of polynomial expansion
    dd = d #Degree of polynomial for roots definition
    NumberOfDataPoints = length(Data)

    m = zeros(T, 2 * dd + 2)
    @batch for col in axes(Data, 2)
        compute_moments!(m, view(Data, :, col), NumberOfDataPoints, dd)
    end
    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1) # Allocate once for all :)
    PolyCoeff_NonNorm_prealloc = zeros(T, dd + 1, dd + 1) # Allocate once for all :)

    for degree in 0:dd
        Hankel = @views OrthogonalBasis[1:(degree + 1), 1:(degree + 1)]
        Vc = zeros(T, degree + 1)
        PolyCoeff_NonNorm = @views PolyCoeff_NonNorm_prealloc[1:(degree + 1), 1:(degree + 1)]

        for i in 0:(degree - 1)
            for j in 0:degree
                Hankel[i + 1, j + 1] = @views m[i + j + 1]  # put in the moment
            end
            Hankel[i + 1, :] = @views Hankel[i + 1, :] / maximum(abs.(@views Hankel[i + 1, :]))
        end
        for j in 0:(degree - 1)
            Hankel[degree + 1, j + 1] = zero(T)
        end
        Hankel[degree + 1, degree + 1] = one(T)
        Hankel[degree + 1, :] = @views Hankel[degree + 1, :] / maximum(abs.(@views Hankel[degree + 1, :]))

        # Loop for Vc
        for i in 0:(degree - 1)
            Vc[i + 1] = zero(T)
        end
        Vc[degree + 1] = one(T)

        PolyCoeff_NonNorm[degree + 1, 1:(degree + 1)] .= Hankel \ Vc

        P_norm = 0.0
        for i in 1:NumberOfDataPoints
            Poly = 0
            for k in 0:degree
                Poly += @views PolyCoeff_NonNorm[degree + 1, k + 1] * Data[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end
        for k in 0:degree
            OrthonormalBasis[degree + 1, k + 1] = @views PolyCoeff_NonNorm[degree + 1, k + 1] / sqrt(P_norm)
        end
    end

    return OrthonormalBasis
end


@stable function aPCE_FullBasis(Data, Degree)
    T = eltype(Data)
    d = Degree #Degree of polynomial expansion
    dd = d #Degree of polynomial for roots definition
    # FullBasis = [i >= j ? one(T) : zero(T) for i in 1:dd+1, j in 1:dd+1]
    # # Do this FullBasis = [i >= j ? one(T) : zero(T) for i in 1:dd+1, j in 1:dd+1] as pre allocated for loop to be type stable
    FullBasis = Matrix{T}(undef, Degree + 1, Degree + 1)

    # Fill the matrix using a type-stable for loop
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


@stable function GaussianCollocation(
        input_dimensions, ExpansionDegree, OrthonormalBasis::AbstractArray{T},
        InputDistribution::AbstractArray{T}, NumberOfTerms; strategy = :PCM
    ) where {T <: Real}
    # @info input_dimensions
    polynomial_roots = zeros(T, input_dimensions, ExpansionDegree + 1)
    @inbounds for d in Base.oneto(Int64(input_dimensions))
        polynomial_basis = @views OrthonormalBasis[:, :, d]
        # @debug "" polynomial_basis
        polynomial_roots[d, :] = @view reinterpret(T, PolynomialRoots.roots(@views polynomial_basis[ExpansionDegree + 2, :]))[1:2:(end - 1)]
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


@stable function numberPolynomials(n, d)
    x, y = max(d, n), min(d, n)
    return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end

@stable function reverse_columns!(x)
    @inbounds for row in axes(x, 1)
        x[row, :] = reverse(@views x[row, :])
    end
    return x
end

function create_basis(x, degree; normalize_data = true)
    return create_basis(x, degree, Val(true); normalize_data = normalize_data)
end

# Univartiate BASIS creation
@stable function create_basis(x, degree, is_orthonormal::Val{true}; normalize_data = true)
    input_dimensions = size(x, 2)
    OrthonormalBasis = Array{eltype(x), 3}(undef, degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_OrthonormalBasis(view(x, :, i), degree, Val(normalize_data))
    end # x[:,i]
    return OrthonormalBasis
end

@stable function create_basis(x, degree, is_orthonormal::Val{false}; normalize_data = true)
    input_dimensions = size(x, 2)
    OrthonormalBasis = Array{eltype(x), 3}(undef, degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_FullBasis(view(x, :, i), degree)
    end
    return OrthonormalBasis
end

@stable function create_basis!(OrthonormalBasis, x, degree; normalize_data = false)
    # @ignore_derivatives begin
    input_dimensions = size(x, 2)
    # OrthonormalBasis = eltype(x).(OrthonormalBasis)
    if eltype(OrthonormalBasis) != eltype(x)
        OrthonormalBasis = eltype(x).(OrthonormalBasis)
    end
    # OrthonormalBasis = zeros(eltype(x), degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_OrthonormalBasis(view(x, :, i), degree, Val(normalize_data))
    end
    # return OrthonormalBasis
    # end
end


@stable function compose_Ψ(x::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis, degree) where {T}
    Ψ = aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis)' |> Matrix{T}
    return Ψ
end


@stable function evaluate_Ψ(x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
    T = eltype(coeffs)
    Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree) #.|> T
    TensorOperations.@tensor PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
    # PredictionOutput = outer_product_kernel(cu(Ψ), cu(coeffs))
    return PredictionOutput
end


@stable function coeffs_from_basis(OrthonormalBasis, degree, ii)
    return OrthonormalBasis[degree, 1:degree, ii]
end

@stable function derivative_coeffs(coeffs)
    if length(coeffs) <= 1
        return [zero(eltype(coeffs))]  # Derivative of a constant polynomial is zero.
    end
    return [i * coeffs[i + 1] for i in 1:(length(coeffs) - 1)]
end


@stable function evalpoly_two(x, cs::AbstractArray)
    i = lastindex(cs)
    out = cs[i]
    i -= 1
    fi = firstindex(cs)
    while i > fi
        out = muladd(out, x, cs[i])
        out = muladd(out, x, cs[i - 1])
        i -= 2
    end
    return i == fi ? muladd(out, x, @inbounds(cs[fi])) : out
end


@stable function evaluate_derivative_horner(x, coeffs)
    n = length(coeffs) - 1
    if n == 0
        return 0.0  # The derivative of a constant polynomial is 0
    end
    derivative_coeffs = [i * coeffs[i + 1] for i in 1:n]  # Compute coefficients for the derivative
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

@stable @inline function evaluate_polynomial_horner_array(x, coeffs)
    results = Vector(undef, length(x))
    for (i, xi) in enumerate(x)
        result = 0.0
        @simd for coeff in reverse(coeffs)
            result = result * xi + coeff
        end
        results[i] = result
    end
    return results
end

@stable function train(Ψ::AbstractArray{T}, y_rhs; bayesian_inversion = :true, reg_order = 0) where {T <: Real}
    NumberOfTerms = size(Ψ, 2)
    output_dimensions = size(y_rhs, 2)
    coeffs = zeros(T, NumberOfTerms, output_dimensions)
    # Ψ = Matrix{T}(Ψ)
    # display(UnicodePlots.spy(sparse(Ψ)))
    Psi_inv = pinv(Ψ; rtol = sqrt(eps(real(float(oneunit(eltype(Ψ)))))))
    @einsum coeffs[i, k] = Psi_inv[i, j] * y_rhs[j, k] # tensoropt einsum

    # for k in axes(y_rhs, 2)
    # 	@info "using rtls" k
    # 	coeffs[:, k] .= rtls(Ψ,y_rhs[:,k])
    # end

    if bayesian_inversion
        @info "Using bayesian regularization y_rhs find the expansion coefficients"
        x₀ = coeffs # Quite a good first guess :) And pinv is
        for i in axes(y_rhs, 2) # stride=true
            @info "Bayesian regularization for axis $i"
            coeffs[:, i] .= invert(Ψ, y_rhs[:, i], Lₖx₀(reg_order, view(x₀, :, i)); alg = :gcv_svd, method = LBFGS(linesearch = LineSearches.BackTracking()))
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
