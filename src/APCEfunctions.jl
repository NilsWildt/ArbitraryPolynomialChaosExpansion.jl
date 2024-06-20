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
# using Hyperopt
# using IterativeSolvers
# using KernelFunctions
# using NonlinearSolve
# using Preconditioners
# using SparseArrays
# using StaticArrays
using StaticArrays
import Optim: NewtonTrustRegion, Options, optimize, minimizer, minimum, LBFGS, IPNewton
using RegularizationTools
import RegularizationTools: Lₖx₀, solve, RegularizationProblem, setupRegularizationProblem, to_general_form, to_standard_form, gcv_tr, gcv_svd, invert, Lₖ, NelderMead, LₖB, Lₖx₀B, LₖDₓ, Lₖx₀Dₓ, LₖDₓB, Lₖx₀DₓB
# using TotalLeastSquares
using Combinatorics
# using OMEinsum
# using CUDA
# using Enzyme
using InducingPoints
using LazyArrays
using LazyGrids
using LinearAlgebra
using LineSearches
using Estrin
using MKL
# using BLISBLAS
using Polyester
using PolynomialRoots
using OnlineStats
using Polynomials
using PrettyTables
using SparseArrays
using StatsBase
using Strided
using TensorOperations
using UnicodePlots # To use spy from SparseArrays
using ChainRulesCore
using UnrolledUtilities
using DispatchDoctor: @stable
# using FastLevenbergMarquardt
# using DifferentiableFactorizations
# using BackwardsLinalg
using ReverseDiff
using ForwardDiff
using TimerOutputs
using Tracker
using StatsBase
# using StaticArrays
using SparseArrays

using LinearAlgebra: svd, norm, pinv, Diagonal, tr
using LinearAlgebra: checksquare
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!
# using FastBroadcast
# using AbstractDifferentiation: AbstractDifferentiation
# const AD = AbstractDifferentiation
using CPUSummary
# using Tapir
# using Enzyme
using Einsum
# using DifferentiationInterface
BLAS.set_num_threads(CPUSummary.get_cpu_threads() ÷ 2)
using Tullio
using Infiltrator


export create_basis!, evaluate_Ψ_zygote, GaussianCollocation, evaluate_Ψ!
@info "Benchmarking Matrix mutplication speed" LinearAlgebra.peakflops(; parallel = true)

Strided.set_num_threads(CPUSummary.get_cpu_threads() ÷ 2)
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
	struct CoSorterElement{T1,T2,T3}
	    x::T1
		z::T2
	    y::T3
	end
	
	struct CoSorter{T1,T2,T3,A <: AbstractVecOrMat{T1},B <: AbstractVecOrMat{T2},C <: AbstractVecOrMat{T3}} <: AbstractVector{CoSorterElement{T1,T2,T3}}
	    sortarray::A
	    otherarray::B
	    coarray::C
	end
	
	Base.size(c::CoSorter) = size(c.sortarray)
	Base.getindex(c::CoSorter, i...) = 
	    CoSorterElement(getindex(c.sortarray, i...), getindex(c.otherarray, i...),getindex(c.coarray, i...))
	Base.setindex!(c::CoSorter, t::CoSorterElement, i...) = 
	    (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c) 

 Base.isless(a::CoSorterElement, b::CoSorterElement) =isless(a.x, b.x) || (a.x == b.x && isless(a.z, b.z))
		
    
	Base.Sort.defalg(v::C) where {T <: Union{Number,Missing},C <: CoSorter{T}} = 
	    Base.DEFAULT_UNSTABLE
	@stable function sort_two_arrays!(x::AbstractArray, y::AbstractArray)
	    T = CoSorter(x[:,1],x[:,2], y)
	    sort!(T)
	    x = T.sortarray
	    y = T.coarray
	end

	@stable function aPCE_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T, s_marginals::F, s_interactions::F)::Matrix{T} where {T<:Integer,F<:Real}
		# Initialize the indices for the first parameter
		@stable function get_stats(r)::Array{F} # Returns sum, nzeros, mean, var, min,max
			o = Series(Mean(),Variance(), Extrema())
			n = length(r)
			summe = 0
			n_zeros = 0
			@inbounds for e in 1:n
				if iszero(r[e])
					n_zeros += 1
				else
					summe += r[e]
				end
				fit!(o,r[e])
			end
			meanval = summe/n
			meanval,varval,mm = value(o)
			return [summe,n_zeros,meanval,varval,mm.min,mm.max]
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

	
		range_ = 0:max_degree |> collect
		indices = reshape(range_, :, 1)  # Make it a column vector
		@inbounds for di in 1:num_dimensions-1
			indices = repeat(indices, inner = (max_degree + 1, 1))
			front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di)
			indices = hcat(front,indices)
			indices = indices[vec(sum(indices; dims = 2)).<=max_degree, :]
		end

		stats = reduce(hcat,map(x->get_stats(x),eachrow(indices)))'
		d_marginal_indices = T[]
		d_interactions_indices = T[]
		@inbounds for r in axes(stats,1) # Go over columns
			if stats[r,1] <= max_degree
				if stats[r,2] == (num_dimensions - 1)
						push!(d_marginal_indices,r)
				elseif (num_dimensions - stats[r,2]) >= 1
							push!(d_interactions_indices,r)
					end
				end
		end
		sorting_d_marginal = stats[d_marginal_indices,3:4] 
		sorting_d_interactions = stats[d_interactions_indices,3:4]
		all_marginals= 1:length(d_marginal_indices) |> collect
		all_interactions = 1:length(d_interactions_indices)|> collect
		if length(all_marginals) > 1
			sort_two_arrays!(sorting_d_marginal,all_marginals)
		end
		if length(all_interactions) > 1
			sort_two_arrays!(sorting_d_interactions,all_interactions)
		end
		keeper_marginals = d_marginal_indices[filter_by_percentage(all_marginals,s_marginals)]
		keeper_interactions = d_interactions_indices[filter_by_percentage(all_interactions,s_interactions)]
		idxkeep = vcat(keeper_marginals,keeper_interactions)
		indices = @views indices[idxkeep,:]
		indices = vcat(indices,zeros(T,num_dimensions)')
		indices = sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
		return indices
	end


@stable function aPCE_PsiPolynomialMatrix_zygote(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis) where {T <: Real}
	NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = Zygote.bufferfrom(ones(eltype(TrainingInput), NumberOfTerms, NCpoints))
	# OrthonormalBasis = T.(OrthonormalBasis)
	# Function to evaluate polynomials for a given term and input sample
	@inbounds for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		# product = 1.0  # Initialize the product for this term and sample
		for ii ∈ 1:InputDimensions  # For each dimension of the input
			degree = MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
			coeffs = OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
			# p = Polynomials.Polynomial{T}(coeffs)  # Create the polynomial
			# p = Poly(coeffs)
			for j ∈ 1:NCpoints  # For each input sample
				x = TrainingInput[j, ii]
				Psi[i, j] *= evalpoly_two(x, coeffs)
				# Psi[i,j] *= evalpoly(x, p)
			end
		end
	end
	return copy(Psi)
end

@stable function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{T}) where {T <: Real}
	NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = ones(eltype(TrainingInput), NumberOfTerms, NCpoints)
	# OrthonormalBasis = T.(OrthonormalBasis)
	# Function to evaluate polynomials for a given term and input sample
	@inbounds for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		# product = 1.0  # Initialize the product for this term and sample
		for ii ∈ 1:InputDimensions  # For each dimension of the input
			degree = @views MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
			coeffs = @views OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
			# p = Polynomials.Polynomial{T}(coeffs)  # Create the polynomial
			p = Poly(coeffs)
			@batch for j ∈ 1:NCpoints  # For each input sample # @batch
				x = @views TrainingInput[j, ii]
				Psi[i, j] *= p(x)  # Evaluate the polynomial at x and multiply
				# Psi[i,j] *= evalpoly(x, p)
			end
		end
	end
	return Psi
end

# Need this function for Orthonormal Basis is a ForwardDiff.
@stable function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{S}) where {S, T <: Real}
	NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = ones(eltype(TrainingInput), NumberOfTerms, NCpoints)
	# OrthonormalBasis = T.(OrthonormalBasis)
	# Function to evaluate polynomials for a given term and input sample
	@inbounds for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		# product = 1.0  # Initialize the product for this term and sample
		for ii ∈ 1:InputDimensions  # For each dimension of the input
			degree = @views MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
			coeffs = @views OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
			p = Polynomials.Polynomial{T}(coeffs)  # Create the polynomial this dispatch uses cating.
			@batch for j ∈ 1:NCpoints  # For each input sample @batch
				x = @views TrainingInput[j, ii]
				Psi[i, j] *= p(x)  # Evaluate the polynomial at x and multiply
				# Psi[i,j] *= evalpoly(x, p)
			end
		end
	end
	return Psi
end

@stable function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{T}) where {T <: ForwardDiff.Dual}
	# @polly function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis) where {T <: Number}
	NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = ones(eltype(TrainingInput), NumberOfTerms, NCpoints)

	# Function to evaluate polynomials for a given term and input sample
	@inbounds for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		# product = 1.0  # Initialize the product for this term and sample
		for ii ∈ 1:InputDimensions  # For each dimension of the input
			degree = @views MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
			coeffs = @views OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
			p = Polynomials.Polynomial{T}(coeffs)  # Create the polynomial
			# p  = Poly(coeffs)
			@batch for j ∈ 1:NCpoints  # For each input sample @batch
				x = @views TrainingInput[j, ii]
				Psi[i, j] *= p(x)  # Evaluate the polynomial at x and multiply
				# Psi[i,j] *= evalpoly(x, p)
			end
		end
	end
	return Psi
end



@stable function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S; normalize_data = false) where {T <: Real, S <: Integer}
	# T = eltype(Data)
	d = Degree #Degree of polinomial expansion
	dd = d #Degree of polinomial for roots defenition
	NumberOfDataPoints = length(Data)
	# VarOfData = var(Data)
	# MeanOfData = 0.0
	if normalize_data
		MeanOfData = mean(Data)
		Data = Data ./ MeanOfData
	end
	m = Zygote.bufferfrom(zeros(T, 2 * dd + 2))
	for l ∈ 0:(2*dd+1)
		m[l+1] = sum(Data .^ l) / NumberOfDataPoints
	end
	# if any(isnan, m)
	# 	@warn "NaNs in the moments"
	# 	@info "" Data NumberOfDataPoints 
	# end
	OrthonormalBasis = Zygote.bufferfrom(zeros(T, dd + 1, dd + 1))
	OrthogonalBasis = Zygote.bufferfrom(zeros(T, dd + 1, dd + 1))
	@inbounds for degree ∈ 0:dd
		Hankel = @views OrthogonalBasis[1:degree+1, 1:degree+1]
		Vc = zeros(T, degree + 1)
		PolyCoeff_NonNorm = copy(Hankel)
		for i ∈ 0:degree
			@batch for j ∈ 0:degree # @batch
				if i < degree
					Hankel[i+1, j+1] = @views m[i+j+1] # put in the moment
				elseif (i == degree) && (j < degree)
					Hankel[i+1, j+1] = 0.0
				elseif (i == degree) && (j == degree)
					Hankel[i+1, j+1] = 1.0
				end
			end
			# fr1 = copy(Hankel); # Control Hankel only considering the raw moments without division by max(abs) in each row
			Hankel[i+1, :] = @views Hankel[i+1, :] / maximum(abs.(@views Hankel[i+1, :]))
		end
		for i ∈ 0:degree
			if (i < degree)
				Vc[i+1] = 0
			elseif (i == degree)
				Vc[i+1] = 1
			end
		end
		Vp = zeros(T, size(Vc))
		try
			Vp .= Hankel \ Vc
		catch
			@warn "Hankel matrix singular, trying pseudo inverse." #  Vp Hankel Vc
			Vp .= pinv(Hankel) * Vc
		end
		# Vp = Hankel \ Vc
		PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Vp
		# @ignore_derivatives begin
		deviation = 100 * abs(sum(abs.(Hankel * PolyCoeff_NonNorm[degree+1, 1:degree+1])) - sum(abs.(Vc)))
		ChainRulesCore.ignore_derivatives() do
			if (deviation > 0.5)
				@warn "Computational error of the linear solver is too high: $(round(deviation;digits=3))"
			end
		end
		#Normalization of polynomial coefficients
		P_norm = 0
		@inbounds for i ∈ 1:NumberOfDataPoints
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
	if normalize_data
		@inbounds for k ∈ 1:lastindex(OrthonormalBasis, 2)
			OrthonormalBasis[:, k] = @views OrthonormalBasis[:, k] ./ (MeanOfData^(k - 1))
		end
	end
	return copy(OrthonormalBasis)
end
# @stable function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S; normalize_data = false) where {T <: Real, S <: Integer}
# 	ChainRulesCore.ignore_derivatives() do
# 		# T = eltype(Data)
# 		d = Degree #Degree of polinomial expansion
# 		dd = d #Degree of polinomial for roots defenition
# 		NumberOfDataPoints = length(Data)
# 		# VarOfData = var(Data)
# 		# MeanOfData = 0.0
# 		if normalize_data
# 			MeanOfData = mean(Data)
# 			Data = Data ./ MeanOfData
# 		end
# 		m = zeros(T, 2 * dd + 2)
# 		for l ∈ 0:(2*dd+1)
# 			m[l+1] = sum(Data .^ l) / NumberOfDataPoints
# 		end
# 		# if any(isnan, m)
# 		# 	@warn "NaNs in the moments"
# 		# 	@info "" Data NumberOfDataPoints 
# 		# end
# 		OrthonormalBasis = zeros(T, dd + 1, dd + 1)
# 		OrthogonalBasis = zeros(T, dd + 1, dd + 1) # Allocate once for all :)
# 		@inbounds for degree ∈ 0:dd
# 			Hankel = @views OrthogonalBasis[1:degree+1, 1:degree+1]
# 			Vc = zeros(T, degree + 1)
# 			PolyCoeff_NonNorm = copy(Hankel)
# 			for i ∈ 0:degree
# 				for j ∈ 0:degree # @batch
# 					if i < degree
# 						Hankel[i+1, j+1] = @views m[i+j+1] # put in the moment
# 					elseif (i == degree) && (j < degree)
# 						Hankel[i+1, j+1] = 0.0
# 					elseif (i == degree) && (j == degree)
# 						Hankel[i+1, j+1] = 1.0
# 					end
# 				end
# 				# fr1 = copy(Hankel); # Control Hankel only considering the raw moments without division by max(abs) in each row
# 				Hankel[i+1, :] = @views Hankel[i+1, :] / maximum(abs.(@views Hankel[i+1, :]))
# 			end
# 			for i ∈ 0:degree
# 				if (i < degree)
# 					Vc[i+1] = 0
# 				elseif (i == degree)
# 					Vc[i+1] = 1
# 				end
# 			end
# 			Vp = zeros(T, size(Vc))
# 			try
# 				Vp .= Hankel \ Vc
# 			catch
# 				@warn "Hankel matrix singular, trying pseudo inverse." #  Vp Hankel Vc
# 				Vp .= pinv(Hankel) * Vc
# 			end
# 			# Vp = Hankel \ Vc
# 			PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Vp
# 			# @ignore_derivatives begin
# 			deviation = 100 * abs(sum(abs.(Hankel * PolyCoeff_NonNorm[degree+1, 1:degree+1])) - sum(abs.(Vc)))
# 			if (deviation > 0.5)
# 				@warn "Computational error of the linear solver is too high: $(round(deviation;digits=3))"
# 			end
# 			#Normalization of polynomial coefficients
# 			P_norm = 0
# 			@inbounds for i ∈ 1:NumberOfDataPoints
# 				Poly = 0
# 				for k ∈ 0:degree
# 					Poly += @views PolyCoeff_NonNorm[degree+1, k+1] * Data[i]^k
# 				end
# 				P_norm += Poly^2 / NumberOfDataPoints
# 			end
# 			for k ∈ 0:degree
# 				OrthonormalBasis[degree+1, k+1] = @views PolyCoeff_NonNorm[degree+1, k+1] / sqrt(P_norm)
# 			end
# 		end
# 		if normalize_data
# 			@inbounds for k ∈ 1:lastindex(OrthonormalBasis, 2)
# 				OrthonormalBasis[:, k] = @views OrthonormalBasis[:, k] ./ (MeanOfData^(k - 1))
# 			end
# 		end
# 		return OrthonormalBasis
# 	end
# end


@stable function GaussianCollocation(input_dimensions, ExpansionDegree, OrthonormalBasis::AbstractArray{T},
	InputDistribution::AbstractArray{T}, NumberOfTerms; strategy = :PCM) where {T <: Real}
	# @info input_dimensions
	polynomial_roots = zeros(T, input_dimensions, ExpansionDegree + 1)
	@inbounds for d ∈ Base.oneto(Int64(input_dimensions))
		polynomial_basis = @views OrthonormalBasis[:, :, d]
		# @debug "" polynomial_basis
		polynomial_roots[d, :] = @view reinterpret(T, PolynomialRoots.roots(@views polynomial_basis[ExpansionDegree+2, :]))[1:2:end-1]
	end
	PointsVector = 1:ExpansionDegree+1 |> collect
	UniqueCombinations = stack(reduce(vcat, (Iterators.product([PointsVector for _ in 1:input_dimensions]...))))'

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
		collocation_points = @strided sortslices(collocation_points, dims = 1, by = x -> x[1])
		return Array(view(collocation_points, :, (1:size(collocation_points, 2))))
	end
end

function KMeansCollocation(InputDistribution, M = 10)
	alg = InducingPoints.KmeansAlg(M)
	Z = inducingpoints(alg, reduce(hcat, InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function kDPPCollocation(InputDistribution, M = 10)
	kernel = SqExponentialKernel()
	alg = kDPP(M)
	Z = inducingpoints(alg, reduce(hcat, InputDistribution)'; kernel)
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function RandomSubsetCollocation(InputDistribution, M = 10)
	alg = RandomSubset(M)
	Z = inducingpoints(alg, reduce(hcat, InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function CoverTreeCollocation(InputDistribution, c = 0.2)
	alg = CoverTree(c)
	Z = inducingpoints(alg, reduce(hcat, InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function UniGridCollocation(InputDistribution, M = 10)
	alg = UniGrid(M)
	Z = inducingpoints(alg, reduce(hcat, InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
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

@stable function create_basis(x, degree; normalize_data = true)
	@ignore_derivatives begin
		input_dimensions = size(x, 2)
		OrthonormalBasis = zeros(eltype(x), degree + 1, degree + 1, input_dimensions)
		@inbounds for i in 1:input_dimensions
			tmp = aPCE_OrthonormalBasis(x[:, i], degree; normalize_data = normalize_data)
			OrthonormalBasis[:, :, i] = tmp
		end
		return OrthonormalBasis
	end
end

@stable function create_basis!(OrthonormalBasis, x, degree; normalize_data = false)
	@ignore_derivatives begin
		input_dimensions = size(x, 2)
		# OrthonormalBasis = eltype(x).(OrthonormalBasis)
		if eltype(OrthonormalBasis) != eltype(x)
			OrthonormalBasis = eltype(x).(OrthonormalBasis)
		end
		# OrthonormalBasis = zeros(eltype(x), degree + 1, degree + 1, input_dimensions)
		@inbounds for i in 1:input_dimensions
			tmp = aPCE_OrthonormalBasis(x[:, i], degree; normalize_data = normalize_data)
			OrthonormalBasis[:, :, i] = tmp
		end
		return OrthonormalBasis
	end
end


@stable function compose_Ψ(x::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis, degree) where T
	Ψ = aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis)' |> Matrix{T}
	return Ψ
end

@stable function compose_Ψ_zygote(x::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis, degree) where T
	Ψ = aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, OrthonormalBasis)' |> Matrix{T}
	return Ψ
end

@stable function evaluate_Ψ_zygote(x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
	T = eltype(coeffs)
	Ψ = compose_Ψ_zygote(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@tensoropt PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
	# @info "" typeof(x) typeof(coeffs) typeof(MultivariatePolynomialDegrees) typeof(OrthonormalBasis) typeof(degree) typeof(name)
	return T.(PredictionOutput)
end

@stable function evaluate_Ψ!(PredictionOutput, x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
	# T = eltype(coeffs)
	Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@tensoropt PredictionOutput[k, j] = Ψ[k, i] * coeffs[i, j]
end


@stable function evaluate_Ψ(x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
	T = eltype(coeffs)
	Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@tullio PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
	# @info "" typeof(x) typeof(coeffs) typeof(MultivariatePolynomialDegrees) typeof(OrthonormalBasis) typeof(degree) typeof(name)
	return T.(PredictionOutput)
end


@stable function evaluate_Ψ(x, coeffs::ReverseDiff.TrackedArray, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
	Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@einsum PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
	return PredictionOutput
end

@stable function evaluate_Ψ(x, coeffs::AbstractVecOrMat{T}, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name) where {T <: ForwardDiff.Dual}
	Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@einsum PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
	return PredictionOutput
end

@stable function evaluate_Ψ(x, coeffs::Tracker.TrackedArray, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
	Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@einsum PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
	return PredictionOutput
end


@stable function coeffs_from_basis(OrthonormalBasis, degree, ii)
	return OrthonormalBasis[degree, 1:degree, ii]
end

@stable function derivative_coeffs(coeffs)
	if length(coeffs) <= 1
		return [zero(eltype(coeffs))]  # Derivative of a constant polynomial is zero.
	end
	return [i * coeffs[i+1] for i in 1:length(coeffs)-1]
end


@stable function evalpoly_two(x, cs::AbstractArray)
	i = lastindex(cs)
	out = cs[i]
	i -= 1
	fi = firstindex(cs)
	while i > fi
		out = muladd(out, x, cs[i])
		out = muladd(out, x, cs[i-1])
		i -= 2
	end

	return i == fi ? muladd(out, x, @inbounds(cs[fi])) : out
end


@stable function evaluate_derivative_horner(x, coeffs)
	n = length(coeffs) - 1
	if n == 0
		return 0.0  # The derivative of a constant polynomial is 0
	end
	derivative_coeffs = [i * coeffs[i+1] for i in 1:n]  # Compute coefficients for the derivative
	if isempty(derivative_coeffs)
		return 0.0
	end
	# Apply Horner's method
	derivative_value = derivative_coeffs[end]
	@simd for i in (n-1):-1:1
		derivative_value = derivative_value * x + derivative_coeffs[i]
	end
	if isnan(derivative_value) || isinf(derivative_value)
		@warn "NaN or Inf in the derivative"
		@infiltrate
	end
	return derivative_value
end

@stable function evaluate_polynomial_horner_array(x, coeffs)
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

@stable function lsqnonneg(C::Matrix, d::Vector, tol::Real = -1, itmax_factor::Real = 3)
	# Set the tolerance
	(m, n) = size(C)
	tol = (tol == -1) ? 10 * eps() * norm(C, 1) * (maximum(size(C)) + 1) : tol
	itmax = itmax_factor * n
	# Initialize vector of n zeros and Infs (to be used later)
	wz = zeros(n)
	# Initialize set of non-active columns to null
	P = falses(n)
	# Initialize set of active columns to all and the initial point to zeros
	Z = trues(n)
	x = zeros(n)
	Ctrans = transpose(C)
	resid = d - C * x
	w = Ctrans * resid
	# Set up iteration criterion
	outeriter = 0
	iter = 0
	exitflag = 1
	# Outer loop to put variables into set to hold positive coefficients
	@inbounds while any(Z) && any(w[Z] .> tol)
		# print("On iteration $(outeriter)\n")
		outeriter += 1
		# Reset intermediate solution z
		z = zeros(n)
		# Create wz, a Lagrange multiplier vector of variables in the zero set.
		# wz must have the same size as w to preserve the correct indices, so
		# set multipliers to -Inf for variables outside of the zero set.
		wz[P] .= -Inf  # Use broadcasting here
		wz[Z] .= w[Z]  # Use broadcasting here
		# Find variable with largest Lagrange multiplier
		t = argmax(wz)
		# Move variable t from zero set to positive set
		P[t] = true
		Z[t] = false
		# Compute intermediate solution using only variables in positive set
		z[P] = C[:, findall(P)] \ d  # Use findall to get column indices
		# Inner loop to remove elements from the positive set which no longer belong
		while any(z[P] .<= 0)
			# print("entering inner loop\n")
			iter += 1
			if iter > itmax
				println("lsqnonneg: IterationCountExceeded")
				exitflag = 0
				iterations = outeriter
				resnorm = sum(resid .* resid)
				x = z
				lambda = w
				return x
			end
			# Find indices where intermediate solution z is approximately negative
			Q = (z .<= 0) .& P
			# Choose new x subject to keeping new x nonnegative
			alpha = minimum(x[Q] ./ (x[Q] .- z[Q]))
			x .= x .+ alpha .* (z .- x)  # Use broadcasting here
			# Reset Z and P given intermediate values of x
			Z .= ((abs.(x) .< tol) .& P) .| Z  # Use broadcasting here
			P .= .~Z  # Use broadcasting here
			z = zeros(n)        # Reset z
			z[P] = C[:, findall(P)] \ d     # Re-solve for z using findall
		end
		x = z
		resid = d - C * x
		w = Ctrans * resid
	end
	return x
end
