### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ef446ee6-3065-452c-8dc5-762b864a8d10
begin
	    import Pkg
	    # activate the shared project environment
	    # Pkg.activate(Base.current_project())
	    Pkg.activate("/data/homes/wildt/Projects/APCE.jl")
	    # instantiate, i.e. make sure that all packages are downloaded
		Pkg.resolve()
	    # Pkg.instantiate()
end

# ╔═╡ 6d37e6ae-5a39-11ef-2591-e98200463156
begin
	using DispatchDoctor
	using LinearAlgebra
	using Random
	using Chairmarks
	using MakieThemes
	using ForwardDiff
	using Polyester
	using OnlineStats: OnlineStats, Extrema, Mean, Series, Variance, eachrow, value
	using Statistics
	using CUDA
	using KernelAbstractions
	using DispatchDoctor
	using Tullio
	using Zygote
	using DataFrames
	using PlutoUI
	using DrWatson
	using MAT
	using Chairmarks
	using Random
	using DifferentiationInterface
	using Einsum
	using TimerOutputs
	using Statistics
	# using Enzyme
	using ForwardDiff
	using TimerOutputs
	using ReverseDiff
	using ChainRulesCore
	using StatsBase: StatsBase, fit!, mean, sum
	using OMEinsum
	 using PrettyTables
	using Bumper
	using Polynomials
	using LazyArrays
	using FastBroadcast
	using UnicodePlots
	using TensorOperations: TensorOperations, @tensoropt, @tensor, @butensor
	# import .EnzymeRules
	using SparseArrays
	# using .EnzymeRules
	# using Plots
	using Krylov
	using AlgebraOfGraphics
	using CairoMakie
	using IterativeSolvers
end

# ╔═╡ 7ea8a787-bc90-4682-88cb-5bd3d2320dd4
begin
	# Pkg.rm("Plots")
	Pkg.add("MakieThemes")
	# Pkg.add("TensorOperations")
	# Pkg.precompile()
	# Pkg.add("Krylov")
end

# ╔═╡ 661d360e-f23d-4452-85d3-f7d701717b50
const to = TimerOutput()

# ╔═╡ 03044526-09db-434f-a4fd-6061b2d115a5
begin
	@stable function normalization_functions(matrix)
	    # Calculate mean and std for each column
	    col_means = StatsBase.mean(matrix, dims=1)
	    col_stds = StatsBase.std(matrix, dims=1)
	    # Define the normalization function
	    normalize = (x) -> (x .- col_means) ./ col_stds
	    # Define the inverse normalization function
	    inverse_normalize = (x) -> x .* col_stds .+ col_means
	    return normalize, inverse_normalize
	end
	
	# Sort mean and variance at same time
	struct SpecialCoSorterElement{T1,T2,T3}
	    x::T1
	    z::T2
	    y::T3
	end
	
	struct CoSorter{T1,T2,T3,A<:AbstractVecOrMat{T1},B<:AbstractVecOrMat{T2},C<:AbstractVecOrMat{T3}} <: AbstractVector{SpecialCoSorterElement{T1,T2,T3}}
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
	
	
	Base.Sort.defalg(v::C) where {T<:Union{Number,Missing},C<:CoSorter{T}} =
	    Base.DEFAULT_UNSTABLE
	@stable function special_sort_two_arrays!(x::AbstractArray, y::AbstractArray)
	    T = CoSorter(x[:, 1], x[:, 2], y)
	    sort!(T)
	    x = T.sortarray
	    y = T.coarray
	end
	
	
	@stable function aPCE_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T, s_marginals::F, s_interactions::F) where {T<:Integer,F<:Real}
	    # Initialize the indices for the first parameter
	    function get_stats(r)::Array{F} # Returns sum, nzeros, mean, var, min,max
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
	
	    @stable function filter_by_percentage(array::AbstractArray, percentage)
	        # Ensure the percentage is within the valid range
	        if percentage < 0.0 || percentage > 1.0
	            throw(ArgumentError("Percentage must be between 0 and 1"))
	        end
	        n = length(array)
	        num_to_keep = round(Int, percentage * n)
	        return @views array[1:num_to_keep]
	    end
	
	
	    # range_ = 0:max_degree |> collect
	    # indices = reshape(range_, :, 1)  # Make it a column vector
	    # @inbounds for di in 1:num_dimensions-1
	    #     indices = repeat(indices, inner=(max_degree + 1, 1))
	    #     front = repeat(range_, outer=div(lastindex(indices), (max_degree + 1)) ÷ di)
	    #     indices = hcat(front, indices)
	    #     indices = indices[vec(sum(indices; dims=2)).<=max_degree, :]
	    # end
	    # range_ = 0:max_degree |> sparse
	    # indices = reshape(range_, :, 1)  
	    # for di in 1:num_dimensions-1
	    #     indices = repeat(indices, inner=(max_degree + 1, 1))
	    #     front = repeat(range_, outer=div(lastindex(indices), (max_degree + 1)) ÷ di)
	    #     indices = hcat(front, indices)
	    #     indices = indices[vec(sum(indices; dims=2)).<=max_degree, :]
	    # end
	
	    range_ = 0:max_degree  # |> sparse
	    indices = reshape(range_, :, 1)
	    for di in 1:num_dimensions-1
	        indices = repeat(indices, inner=(max_degree + 1, 1))
	        front = repeat(range_, outer=div(lastindex(indices), (max_degree + 1)) ÷ di) |> sparse
	        indices = ApplyArray(hcat, front, indices)
	        indices = @~ indices[vec(sum(indices; dims=2)).<=max_degree, :]
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
	    indices = sortslices(hcat(vec(sum(indices; dims=2)), indices); dims=1, rev=false)[:, 2:end]
	    # reverse_columns!(indices)
	    return indices::Matrix{T}# from sparse to matrix.
	end
	# Need this function for Orthonormal Basis is a ForwardDiff.
@stable function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{S}) where {S,T<:Real}
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = ones(T, NumberOfTerms, NCpoints)
    # OrthonormalBasis = T.(OrthonormalBasis)
    # Function to evaluate polynomials for a given term and input sample
    @inbounds for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
        # product = 1.0  # Initialize the product for this term and sample
        for ii ∈ 1:InputDimensions  # For each dimension of the input
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

	
@stable @inline function compute_moments!(m::AbstractArray{T}, Data::AbstractArray{T}, NumberOfDataPoints, dd) where {T<:Real}
    current_power = ones(T, length(Data))  # Start with Data .^ 0 which is 1
    @inbounds for l ∈ 0:(2*dd+1)
        m[l+1] = sum(current_power) / NumberOfDataPoints
        current_power .*= Data  # Increment the power of Data
    end
end


 @inbounds function aPCE_OrthonormalBasis(Data, Degree::S, normalize_data::Val{true}) where {S<:Integer}
    T = eltype(Data)
    d = Degree #Degree of polinomial expansion
    dd = d #Degree of polinomial for roots defenition
    # @warn "Weird, ist that properly normalized? Not /std and minus mean?"
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

    for degree ∈ 0:dd
        Hankel = @views OrthogonalBasis[1:degree+1, 1:degree+1]
        Vc = zeros(T, degree + 1)
        PolyCoeff_NonNorm = @views PolyCoeff_NonNorm_prealloc[1:degree+1, 1:degree+1]
        # for i ∈ 0:degree
        #     for j ∈ 0:degree # @batch
        #         if i < degree
        #             Hankel[i+1, j+1] = @views m[i+j+1] # put in the moment
        #         elseif (i == degree) && (j < degree)
        #             Hankel[i+1, j+1] = zero(T)
        #         elseif (i == degree) && (j == degree)
        #             Hankel[i+1, j+1] = one(T)
        #         end
        #     end
        #     Hankel[i+1, :] = @views Hankel[i+1, :] / maximum(abs.(@views Hankel[i+1, :]))
        # end
        for i in 0:degree-1
            for j in 0:degree
                Hankel[i+1, j+1] = @views m[i+j+1]  # put in the moment
            end
            Hankel[i+1, :] = @views Hankel[i+1, :] / maximum(abs.(@views Hankel[i+1, :]))
        end
        for j in 0:degree-1
            Hankel[degree+1, j+1] = zero(T)
        end
        Hankel[degree+1, degree+1] = one(T)
        Hankel[degree+1, :] = @views Hankel[degree+1, :] / maximum(abs.(@views Hankel[degree+1, :]))

        # Loop for Vc
        for i in 0:degree-1
            Vc[i+1] = zero(T)
        end
        Vc[degree+1] = one(T)

        # for i ∈ 0:degree
        #     if (i < degree)
        #         Vc[i+1] = zero(T)
        #     elseif (i == degree)
        #         Vc[i+1] = one(T)
        #     end
        # end
        # Vp = zeros(T, size(Vc))
        # try
        PolyCoeff_NonNorm[degree+1, 1:degree+1] .= LinearAlgebra.factorize(Hankel) \ Vc
        # PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Krylov.usymlq(Hankel,Vc) |> first
        # catch
        #     @warn "Hankel matrix singular, trying pseudo inverse." #  Vp Hankel Vc
        # PolyCoeff_NonNorm[degree+1, 1:degree+1] .= pinv(Hankel) * Vc
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
        for i ∈ 1:NumberOfDataPoints
            Poly = 0
            for k ∈ 0:degree
                @fastmath Poly += @views PolyCoeff_NonNorm[degree+1, k+1] * Data[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end
        for k ∈ 0:degree
            @fastmath OrthonormalBasis[degree+1, k+1] = @views PolyCoeff_NonNorm[degree+1, k+1] / sqrt(P_norm)
        end
    end
    for k ∈ 1:lastindex(OrthonormalBasis, 2)
        @fastmath OrthonormalBasis[:, k] = @views OrthonormalBasis[:, k] ./ (MeanOfData^(k - 1))
    end
    return OrthonormalBasis
end


@inbounds function aPCE_OrthonormalBasis(Data, Degree::S, normalize_data::Val{false}) where {S<:Integer}
    T = eltype(Data)
    d = Degree #Degree of polinomial expansion
    dd = d #Degree of polinomial for roots defenition
    NumberOfDataPoints = length(Data)

    m = zeros(T, 2 * dd + 2)
    @batch for col in axes(Data, 2)
        compute_moments!(m, view(Data, :, col), NumberOfDataPoints, dd)
    end
    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1) # Allocate once for all :)
    PolyCoeff_NonNorm_prealloc = zeros(T, dd + 1, dd + 1) # Allocate once for all :)

    for degree ∈ 0:dd
        Hankel = @views OrthogonalBasis[1:degree+1, 1:degree+1]
        Vc = zeros(T, degree + 1)
        PolyCoeff_NonNorm = @views PolyCoeff_NonNorm_prealloc[1:degree+1, 1:degree+1]

        for i in 0:degree-1
            for j in 0:degree
                Hankel[i+1, j+1] = @views m[i+j+1]  # put in the moment
            end
            Hankel[i+1, :] = @views Hankel[i+1, :] / maximum(abs.(@views Hankel[i+1, :]))
        end
        for j in 0:degree-1
            Hankel[degree+1, j+1] = zero(T)
        end
        Hankel[degree+1, degree+1] = one(T)
        Hankel[degree+1, :] = @views Hankel[degree+1, :] / maximum(abs.(@views Hankel[degree+1, :]))

        # Loop for Vc
        for i in 0:degree-1
            Vc[i+1] = zero(T)
        end
        Vc[degree+1] = one(T)

        PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Hankel \ Vc #  pinv(Hankel) * Vc # 

        P_norm = 0.0
        for i ∈ 1:NumberOfDataPoints
            Poly = 0
            for k ∈ 0:degree
                Poly += @views PolyCoeff_NonNorm[degree+1, k+1] * Data[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end
        for k ∈ 0:degree
            OrthonormalBasis[degree+1, k+1] = @views PolyCoeff_NonNorm[degree+1, k+1] / sqrt(P_norm)
        end
    end

    return OrthonormalBasis
end


@stable function create_basis(x, degree,onb=true; normalize_data=true)
    input_dimensions = size(x, 2)
    OrthonormalBasis = Array{eltype(x),3}(undef, degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_OrthonormalBasis(view(x, :, i), degree, Val(normalize_data))
    end
    return OrthonormalBasis
end

	
@stable function create_basis(x, degree,onb=false; normalize_data=true,)
    input_dimensions = size(x, 2)
    OrthonormalBasis = Array{eltype(x),3}(undef, degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_FullBasis(view(x, :, i), degree) 
    end
    return OrthonormalBasis
end

	@stable function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{S}) where {S,T<:Real}
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = ones(T, NumberOfTerms, NCpoints)
    # OrthonormalBasis = T.(OrthonormalBasis)
    # Function to evaluate polynomials for a given term and input sample
    @inbounds for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
        # product = 1.0  # Initialize the product for this term and sample
        for ii ∈ 1:InputDimensions  # For each dimension of the input
            degree = MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
            coeffs = @views OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
            p = Polynomials.Polynomial(coeffs)  # Create the polynomial
            # p = Poly(coeffs)
            x = @views TrainingInput[:, ii]
            @..  thread=true Psi[i, :] *= p(x)  # Evaluate the polynomial at x and multiply
            # Psi[i,j] *= evalpoly(x, p)
        end
    end
    return Psi
end

	@inbounds function aPCE_FullBasis(Data, Degree::S) where {S<:Integer}
	    T = eltype(Data)
	    d = Degree #Degree of polinomial expansion
	    dd = d #Degree of polinomial for roots defenition
		OrthonormalBasis = [i >= j ? one(T) : zero(T) for i in 1:dd+1, j in 1:dd+1]
	    return OrthonormalBasis
	end
end

# ╔═╡ 84fb4fa6-3e31-4f96-95bb-48f808176d45


# ╔═╡ eed0d499-122c-4754-b35c-217d49a17cac

@stable function GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis::AbstractArray{T},
    InputDistribution::AbstractArray{T}, NumberOfTerms; strategy=:PCM) where {T<:Real}
    # @info input_dimensions
    polynomial_roots = zeros(T, input_dimensions, ExpansionDegree + 1)
    @inbounds for d ∈ Base.oneto(Int64(input_dimensions))
        polynomial_basis = @views OrthonormalBasis[:, :, d]
        # @debug "" polynomial_basis
        polynomial_roots[d, :] = @view reinterpret(T, PolynomialRoots.roots(@views polynomial_basis[ExpansionDegree+2, :]))[1:2:end-1]
    end
    PointsVector = 1:ExpansionDegree+1 |> collect
    UniqueCombinations = stack(reduce(vcat, (UnrolledUtilities.unrolled_product([PointsVector for _ in 1:input_dimensions]...))))'

    sort_indices = sortperm(sum(UniqueCombinations; dims=2); dims=1)
    SortUniqueCombinations = UniqueCombinations[sort_indices[:], :]
    if strategy == :FT
        TrainingInput = SortUniqueCombinations
        return Array(view(TrainingInput, :, (1:size(TrainingInput, 2))))
    elseif strategy == :PCM
        temp = abs.(polynomial_roots .- StatsBase.mean(InputDistribution; dims=1)[:, :][1])
        temp_sort = mapslices(sortperm, temp, dims=2)
        @inbounds for i in axes(polynomial_roots, 1)
            polynomial_roots[i, :] = @views polynomial_roots[i, temp_sort[i, :]]
        end
        collocation_points = zeros(T, NumberOfTerms, input_dimensions)
        @inbounds for i in 1:NumberOfTerms
            for j in axes(SortUniqueCombinations, 2)
                collocation_points[i, j] = @views polynomial_roots[j, Int(SortUniqueCombinations[i, j])]
            end
        end
        collocation_points = sortslices(collocation_points, dims=1, by=x -> x[1])
        return Array(view(collocation_points, :, (1:size(collocation_points, 2))))
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

@stable function create_basis(x, degree; normalize_data=true)
    input_dimensions = size(x, 2)
    OrthonormalBasis = Array{eltype(x),3}(undef, degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_OrthonormalBasis(view(x, :, i), degree, Val(normalize_data)) # view(x, :, i)
    end
    return OrthonormalBasis
end

@stable function create_basis!(OrthonormalBasis, x, degree; normalize_data=false)
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
    # local T = eltype(coeffs)
    Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree) #.|> T
    @tensor PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
    # PredictionOutput = outer_product_kernel(cu(Ψ), cu(coeffs))
    return PredictionOutput
end


# @stable function coeffs_from_basis(OrthonormalBasis, degree, ii)
#     return OrthonormalBasis[degree, 1:degree, ii]
# end

# @stable function derivative_coeffs(coeffs)
#     if length(coeffs) <= 1
#         return [zero(eltype(coeffs))]  # Derivative of a constant polynomial is zero.
#     end
#     return [i * coeffs[i+1] for i in 1:length(coeffs)-1]
# end


# @stable function evalpoly_two(x, cs::AbstractArray)
#     i = lastindex(cs)
#     out = cs[i]
#     i -= 1
#     fi = firstindex(cs)
#     while i > fi
#         out = muladd(out, x, cs[i])
#         out = muladd(out, x, cs[i-1])
#         i -= 2
#     end

#     return i == fi ? muladd(out, x, @inbounds(cs[fi])) : out
# end


# @stable function evaluate_derivative_horner(x, coeffs)
#     n = length(coeffs) - 1
#     if n == 0
#         return 0.0  # The derivative of a constant polynomial is 0
#     end
#     derivative_coeffs = [i * coeffs[i+1] for i in 1:n]  # Compute coefficients for the derivative
#     if isempty(derivative_coeffs)
#         return 0.0
#     end
#     # Apply Horner's method
#     derivative_value = derivative_coeffs[end]
#     @simd for i in (n-1):-1:1
#         derivative_value = derivative_value * x + derivative_coeffs[i]
#     end
#     if isnan(derivative_value) || isinf(derivative_value)
#         @warn "NaN or Inf in the derivative"
#     end
#     return derivative_value
# end

# @stable @inline function evaluate_polynomial_horner_array(x, coeffs)
#     results = Vector(undef, length(x))
#     for (i, xi) in enumerate(x)
#         result = 0.0
#         @simd for coeff in reverse(coeffs)
#             result = result * xi + coeff
#         end
#         results[i] = result
#     end
#     return results
# end

# @stable function train(Ψ::AbstractArray{T}, y_rhs; bayesian_inversion=:true, reg_order=0) where {T<:Real}
#     NumberOfTerms = size(Ψ, 2)
#     output_dimensions = size(y_rhs, 2)
#     coeffs = zeros(T, NumberOfTerms, output_dimensions)
#     # Ψ = Matrix{T}(Ψ)
#     # display(UnicodePlots.spy(sparse(Ψ)))
#     Psi_inv = pinv(Ψ; rtol=sqrt(eps(real(float(oneunit(eltype(Ψ)))))))
#     @einsum coeffs[i, k] = Psi_inv[i, j] * y_rhs[j, k] # tensoropt einsum

#     # for k in axes(y_rhs, 2)
#     # 	@info "using rtls" k
#     # 	coeffs[:, k] .= rtls(Ψ,y_rhs[:,k])
#     # end

#     if bayesian_inversion
#         @info "Using bayesian regularization y_rhs find the expansion coefficients"
#         x₀ = coeffs # Quite a good first guess :) And pinv is 
#         for i in axes(y_rhs, 2) # stride=true 
#             @info "Bayesian regularization for axis $i"
#             coeffs[:, i] .= invert(Ψ, y_rhs[:, i], Lₖx₀(reg_order, view(x₀, :, i)); alg=:gcv_svd, method=LBFGS(linesearch=LineSearches.BackTracking()))
#         end
#     end
#     ChainRulesCore.ignore_derivatives() do
#         for k in axes(coeffs, 2)
#             res = (@views sqrt(mean((Ψ * coeffs[:, k] .- y_rhs[:, k]) .^ 2)))
#             @info "Error for axis $k" res
#         end
#     end
#     return coeffs
# end




end

# ╔═╡ 52623f4d-16d4-49c7-bbcc-627aa2691215
@stable function create_basis(x, degree,onb::Val{true}; normalize_data=true)
    input_dimensions = size(x, 2)
    OrthonormalBasis = Array{eltype(x),3}(undef, degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_OrthonormalBasis(view(x, :, i), degree, Val(normalize_data)) 
    end
    return OrthonormalBasis
end

# ╔═╡ bdba7a82-ac0b-41d7-92bf-40ec723099fe
@stable function create_basis(x, degree,onb::Val{false}; normalize_data=true)
    input_dimensions = size(x, 2)
    OrthonormalBasis = Array{eltype(x),3}(undef, degree + 1, degree + 1, input_dimensions)
    for i in 1:input_dimensions
        OrthonormalBasis[:, :, i] .= aPCE_FullBasis(view(x, :, i), degree) 
    end
    return OrthonormalBasis
end

# ╔═╡ c886e031-9ca6-4b96-86fe-eeddbdf35499
function create_basis(x, degree,onb; normalize_data=true)
	return create_basis(x,degree,Val(onb);normalize_data=normalize_data)
end

# ╔═╡ e254e16b-924c-4a5e-9b16-fbcf6f52ebc0
create_basis(rand(3,3),3,true)

# ╔═╡ f1eeedba-acd9-421e-9290-f30754197498
create_basis(rand(3,3),3,false)

# ╔═╡ a5349a70-55d4-4610-8c54-7e29d4315467
function create_vandermonde(x::AbstractArray, max_degree::Int=1)
    n = length(x)
    # Create the Vandermonde matrix
    vandermonde_matrix = [x[i]^(j-1) for i in 1:n, j in 1:(max_degree+1)]
    return vandermonde_matrix
end

# ╔═╡ ce23dde8-3d40-49af-8728-c5784509ac7c
function run_comparison(N,degree,onb)
		FT = Float64
		# N = 100
		

        file = matread("/data/homes/wildt/Projects/LuxApceLayer.jl/data/ReferenceSolution_10P.mat")
        print(keys(file))
        indall = 1:N
        rng = Xoshiro(42)
        # @info "" size(file["TrainingOutput"])
        TrainingInput = file["TrainingInput"][indall, 1] |> Array{FT}
        InputDistributions = file["Input_distributions"][:, 1] |> Array{FT}
        TrainingOutput = file["TrainingOutput"][indall, :] |> Array{FT}
        # @info "" size(TrainingInput)
        ValidationInput = file["Input_distributions"][:, 1] |> Array{FT}
        ValidationOutput = file["ValidationOutput"][:, :] |> Array{FT}

	
	InputDimensions = 10
	outdim = size(TrainingOutput,2)
	
	# degree = 2
	MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(InputDimensions, degree, 1.0, 1.0)
	NumberOfTerms = min(size(MultivariatePolynomialDegrees, 1), numberPolynomials(degree, InputDimensions))  
	OrthonormalBasis = create_basis(InputDistributions, degree,onb; normalize_data=true)
	# UnicodePlots.spy(Matrix(sparse(OrthonormalBasis)))
	# @info "" mean(OrthonormalBasis) 
	Ψ_train = aPCE_PsiPolynomialMatrix(TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)'
	# Psi_inv = pinv(Ψ_train; rtol=sqrt(eps(real(float(oneunit(eltype(Ψ_train)))))))
	# @info "" cond(Ψ_train) 
	Psi_inv = pinv(Ψ_train; rtol=sqrt(eps(real(float(oneunit(eltype(Ψ_train)))))))

	# @info "" size(Psi_inv) size(TrainingOutput)
	coeffs = zeros(FT, NumberOfTerms, outdim)
    @tensor coeffs[i, k] = Psi_inv[i, j] * TrainingOutput[j, k] # tensoropt einsum

	Ψ = aPCE_PsiPolynomialMatrix(ValidationInput, MultivariatePolynomialDegrees, OrthonormalBasis) 
	# @info "" size(Ψ) size(coeffs)
	PredictionOutput = zeros(size(ValidationInput,1),outdim)
	TensorOperations.@tensor PredictionOutput[k, j] = Ψ[i, k] * coeffs[i, j]
	# @info size(ValidationOutput|>vec)
	# @info size(PredictionOutput|>vec)
	UnicodePlots.scatterplot(ValidationOutput|>vec,PredictionOutput|>vec) |> display
	# @info "comparison" mean(ValidationOutput|>vec) mean(PredictionOutput|>vec)


	# Ψ_train = create_vandermonde(TrainingInput)+0.01I
	# Psi_inv = pinv(Ψ_train; rtol=sqrt(eps(real(float(oneunit(eltype(Ψ_train)))))))

end

# ╔═╡ 84a908e5-4950-4401-92a8-d3b21f92b52e
# run_comparison(10,4,false)

# ╔═╡ e897a6a6-68da-4077-a41c-ab80b9e7f356
# run_comparison(100,2,true)

# ╔═╡ 962ba99e-2e94-443e-88e4-9164be1441b6
function obj(α,Vm,Y) 
	Yh = Vm*α
	return sqrt(mean(abs.(Y.-Yh).^2))
end

# ╔═╡ c2be9c6b-a375-4ed9-8e93-71fa75c9300c
function get_V_onb(x,degree)
	InputDimensions = size(x,2)
MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(InputDimensions, degree, 1.0, 1.0)
	NumberOfTerms = min(size(MultivariatePolynomialDegrees, 1), numberPolynomials(degree, InputDimensions))  
	OrthonormalBasis = create_basis(x, degree,Val(true); normalize_data=true)
	# UnicodePlots.spy(Matrix(sparse(OrthonormalBasis)))
	# @info "" mean(OrthonormalBasis) 
	Ψ_train = aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis)'
	return Ψ_train
end

# ╔═╡ e49b643a-f75e-4896-b61f-88ba3ef372c9
function get_V_full(x,degree)
	InputDimensions = size(x,2)
MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(InputDimensions, degree, 1.0, 1.0)
	NumberOfTerms = min(size(MultivariatePolynomialDegrees, 1), numberPolynomials(degree, InputDimensions))  
	OrthonormalBasis = create_basis(x, degree,Val(false); normalize_data=true)
	# UnicodePlots.spy(Matrix(sparse(OrthonormalBasis)))
	# @info "" mean(OrthonormalBasis) 
	Ψ_train = aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis)'
	return Ψ_train
end

# ╔═╡ 311735c7-5df8-4d99-bed9-85a8455b3341
function get_V_vandermonde(x,degree)
return create_vandermonde(x,degree)
end

# ╔═╡ 1015535b-6a87-4b7f-84fc-1b683a52ee04
function add_dim_at_end(arr)
    return reshape(arr, size(arr)..., 1)
end

# ╔═╡ 1371a89b-addf-48b2-aa75-3234c041f25e
# begin
	
# 	# Slider for polynomial degree (1 to 10)
# 	@bind polynomial_degree Slider(1:10, show_value=true, default=3, label="Polynomial Degree")
	
# 	# Slider for the number of samples (10 to 100)
# 	@bind num_samples Slider(10:100, show_value=true, default=50, label="Number of Samples")
# end

# ╔═╡ 441c5bdc-2b6e-4a9a-a6cf-4452c49bb08d


# ╔═╡ 4d48a009-5773-4778-b961-71867593cfd4
function table_create()
	reset_timer!(to)
		FT = Float64
		N = 1000
		# degree = 10
		ps = []
		errors = []
	
		for degree ∈ 1:2:11
		outdim = 1
 #        file = matread("/data/homes/wildt/Projects/LuxApceLayer.jl/data/ReferenceSolution_10P.mat")
 #        # print(keys(file))
 #        indall = 1:N
 #        rng = Xoshiro(42)
 #        # @info "" size(file["TrainingOutput"])
 #        TrainingInput = file["TrainingInput"][indall, 1] |> Array{FT}
	# # @info size(TrainingInput) TrainingInput
		TrainingInput = 5.0.+randn(N,1)#LinRange(0,1,N) |> collect |> add_dim_at_end

		f(x) = x^4+sin(x)+exp(x)
		TrainingOutput = f.(TrainingInput)
        # InputDistributions = file["Input_distributions"][:, 1] |> Array{FT}
        # TrainingOutput = file["TrainingOutput"][indall, :] |> Array{FT}
        # # @info "" size(TrainingInput)
        # ValidationInput = file["Input_distributions"][:, 1] |> Array{FT}
        # ValidationOutput = file["ValidationOutput"][:, :] |> Array{FT}
		V1 = get_V_onb(TrainingInput,degree) 
		V2 = get_V_full(TrainingInput,degree) 
		V3 = get_V_vandermonde(TrainingInput,degree) 

		Vs = Matrix{FT}[V1,V2,V3]
		names = ["onb","full","vandermonde"]
		for (i,V) in enumerate(Vs)
			# @info names[i]
			# display(V)
			# Psi_inv = pinv(V)
			# coeffs = zeros(FT, size(V,1), outdim)
	    	# @tensor coeffs[i, k] := Psi_inv[i, j] * TrainingOutput[j, k]
			# @info "" size(V) size(TrainingOutput) size(coeffs)
			coeffs = undef
			stats = undef
			@timeit to "$(names[i])_backslash" begin
						coeffs = V\TrainingOutput
			end
		
			@timeit to "$(names[i])_lsqr" begin
						coeffs = IterativeSolvers.lsqr(V, TrainingOutput|>vec)
			end
				@timeit to "$(names[i])_cgls" begin
				(coeffs, stats) = cgls(V, TrainingOutput|>vec)
			end
			
			@timeit to "$(names[i])_cgls_itmax_3" begin
				(coeffs, _) = cgls(V, TrainingOutput|>vec;itmax=2)
			end
			
			# @show size(coeffs) stats
			coeffs = add_dim_at_end(coeffs)
			TensorOperations.@tensor PredictionOutput[k, j] := V[k,i] * coeffs[i, j]
			# @info "Deviation in mse" mean((PredictionOutput.-TrainingOutput).^2)
			# @info mean(abs.(PredictionOutput.-TrainingOutput))

			# UnicodePlots.scatterplot(TrainingInput |> vec,log.(abs.(PredictionOutput.-TrainingOutput))|> vec)  |> display
			# push!(ps,p1)
			push!(errors,(;basis=names[i],degree=degree,mse= mean((PredictionOutput.-TrainingOutput).^2),cond=cond(V),iter=stats.niter))
		end
		end
	# Convert the named tuple list to a vector of vectors for pretty_table
# data = [ (err.basis, err.degree, err.mse) for err in errors ]
# data = reduce(hcat,errors)
# # Define the headers
header = ["Basis", "Degree", "MSE","cond","iter"]
# @info "" data
# # Display the table
# 	display(to)
# table = pretty_table(
#     data;header =header)
	
	table = DataFrame(errors,header)
	return table
end

# ╔═╡ 4be5c6de-edcc-4506-b33a-1e6c33051a4f
table = table_create()

# ╔═╡ 7bff53b5-f308-404f-b793-684245ec1571
function normalize_by_group(df::DataFrame, group_col::Symbol, value_col)
    # Convert value_col to Symbol if it is a String
    value_col = String(value_col)
    group_col = String(group_col)

	grouped_df = groupby(df, group_col)
    
    # Normalize each group's values by the group's minimum value
    normalized_values = combine(grouped_df) do subdf
        max_value = maximum(subdf[!, value_col])
        subdf[!, value_col] ./ max_value*100
    end
    
    # Add normalized values as a new column
    df[!, Symbol("Normalized_" * string(value_col))] = normalized_values[!,:x1]
    df
end

# ╔═╡ 8c5898b7-9f4b-4a79-aa1e-d565f3ee71c5
let
	df = DataFrame(table)
	df = normalize_by_group(df, :Degree, :MSE)
	df = normalize_by_group(df, :Degree, :iter)
df
end

# ╔═╡ a55b4658-f4e4-4181-813a-c51fb385a3f5
let
	with_theme(merge(CairoMakie.theme_latexfonts(),MakieThemes.ggthemr(:fresh))) do 
	update_theme!(fontsize=14)
	axis = (width = 600, height = 600)
	df = DataFrame(table)
	df = normalize_by_group(df, :Degree, :MSE)
	df = normalize_by_group(df, :Degree, :iter)
	
	layers =visual(BarPlot, direction = :y, width = 0.8,dodge_gap=0.01)
	d1 = data(df) *  mapping(:Degree,:Normalized_MSE,color=:Basis, dodge = :Basis) *layers
	d2 = data(df) *  mapping(:Degree,:Normalized_iter,color=:Basis, dodge = :Basis) *layers
	# f = Figure()
	# draw!(f[1, 1], d1;axis=(yscale=log,))
	# draw!(f[1, 2], d2)

	f = Figure(figure_padding = 10.0;size=(800,250))
	ga = f[1, 1] 
	gb = f[1, 2] 
	gc = f[1, 3] 
		
		
	ax1 = Axis(ga,ylabel="Normalized MSE in log %",xlabel="Max degree",yscale=log,xticks = 1:2:11)
	ax2 = Axis(gb,ylabel="Normalized iterationsi in log %",xlabel="Max degree",yscale=log,xticks =  1:2:11)

	
	
		
	grid1 = draw!(ax1, d1)
	grid2 = draw!(ax2, d2)

	leg = legend!(gc, grid1;framevisible = false,margin = (-10, -10, -10, -10),
         orientation = :vertical,haligns=:right,valigns=:top,tellheight = false,
        tellwidth = false,padding = (0, 0, 0, 0))

		colsize!(f.layout, 3, Auto(0.3))
		
	f
		save("comparison_basis.png",f)
		save("comparison_basis.pdf",f)
		
		f
	end
end

# ╔═╡ e8f9ae3a-4ef1-4072-90ff-72cb60ab60ec
let
	N = 10 
	degree =3
			TrainingInput = 5.0.+rand(N,1)#LinRange(0,1,N) |> collect |> add_dim_at_end

		f(x) = x^4+sin(x)#+x^12+exp(x)
		TrainingOutput = f.(TrainingInput)
        # InputDistributions = file["Input_distributions"][:, 1] |> Array{FT}
        # TrainingOutput = file["TrainingOutput"][indall, :] |> Array{FT}
        # # @info "" size(TrainingInput)
        # ValidationInput = file["Input_distributions"][:, 1] |> Array{FT}
        # ValidationOutput = file["ValidationOutput"][:, :] |> Array{FT}
		V1 = get_V_onb(TrainingInput,degree)  |> display
		V2 = get_V_full(TrainingInput,degree)   |> display
		V3 = get_V_vandermonde(TrainingInput,degree)   |> display

end

# ╔═╡ Cell order:
# ╠═ef446ee6-3065-452c-8dc5-762b864a8d10
# ╠═7ea8a787-bc90-4682-88cb-5bd3d2320dd4
# ╠═6d37e6ae-5a39-11ef-2591-e98200463156
# ╠═661d360e-f23d-4452-85d3-f7d701717b50
# ╠═03044526-09db-434f-a4fd-6061b2d115a5
# ╠═84fb4fa6-3e31-4f96-95bb-48f808176d45
# ╟─eed0d499-122c-4754-b35c-217d49a17cac
# ╠═52623f4d-16d4-49c7-bbcc-627aa2691215
# ╠═bdba7a82-ac0b-41d7-92bf-40ec723099fe
# ╠═c886e031-9ca6-4b96-86fe-eeddbdf35499
# ╠═e254e16b-924c-4a5e-9b16-fbcf6f52ebc0
# ╠═f1eeedba-acd9-421e-9290-f30754197498
# ╠═a5349a70-55d4-4610-8c54-7e29d4315467
# ╠═ce23dde8-3d40-49af-8728-c5784509ac7c
# ╠═84a908e5-4950-4401-92a8-d3b21f92b52e
# ╠═e897a6a6-68da-4077-a41c-ab80b9e7f356
# ╠═962ba99e-2e94-443e-88e4-9164be1441b6
# ╠═c2be9c6b-a375-4ed9-8e93-71fa75c9300c
# ╠═e49b643a-f75e-4896-b61f-88ba3ef372c9
# ╠═311735c7-5df8-4d99-bed9-85a8455b3341
# ╠═1015535b-6a87-4b7f-84fc-1b683a52ee04
# ╠═1371a89b-addf-48b2-aa75-3234c041f25e
# ╠═441c5bdc-2b6e-4a9a-a6cf-4452c49bb08d
# ╠═4d48a009-5773-4778-b961-71867593cfd4
# ╠═4be5c6de-edcc-4506-b33a-1e6c33051a4f
# ╠═7bff53b5-f308-404f-b793-684245ec1571
# ╠═8c5898b7-9f4b-4a79-aa1e-d565f3ee71c5
# ╠═a55b4658-f4e4-4181-813a-c51fb385a3f5
# ╠═e8f9ae3a-4ef1-4072-90ff-72cb60ab60ec
