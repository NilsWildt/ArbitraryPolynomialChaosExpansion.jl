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
using LinearAlgebra
using RegularizedLeastSquares
using FastLevenbergMarquardt
using Zygote
using NonlinearSolve, StaticArrays
using IterativeSolvers
using Enzyme
using Hyperopt
using Preconditioners
using CUDA
import Optim: NewtonTrustRegion, Options, optimize, minimizer, minimum, LBFGS
import RegularizationTools: Lₖx₀, solve, RegularizationProblem, setupRegularizationProblem, to_general_form, to_standard_form, gcv_tr, gcv_svd, invert, Lₖ, NelderMead
using LazyGrids
using Combinatorics
using PyCall
using InducingPoints
using Polynomials
using KernelFunctions
using StatsBase

mutable struct aPC{T <: Real}
	InputDistribution::RowVecs{T} # in [ d x N-samples]
	input_dimensions::Int64
	ExpansionDegree::Int64
	NumberOfTerms::Int64
	MultivariatePolynomialDegrees::AbstractArray{Int64}
	OrthonormalRepresentation::Bool
	OrthonormalBasis::AbstractArray{T}
	# NumberOfOutputs::Int64
	ExpansionCoefficients::AbstractVector{T}

	# Constructor
	function aPC(
		InputDistribution::RowVecs{T},
		ExpansionDegree::Int64,
		OrthonormalRepresentation::Bool = true,
	) where T
		input_dimensions = size(InputDistribution[1], 1)
		# @info "" size(InputDistribution[1])
		MultivariatePolynomialDegrees = aPC_MultivariatePolynomialDegrees(input_dimensions, ExpansionDegree)
		# display(MultivariatePolynomialDegrees)
		NumberOfTerms = numberPolynomials(ExpansionDegree, input_dimensions)

		OrthonormalBasis = zeros(ExpansionDegree + 2, ExpansionDegree + 2, input_dimensions)
		for i in 1:input_dimensions
			tmp = aPC_OrthonormalBasis(getindex.(InputDistribution, i), ExpansionDegree)
			OrthonormalBasis[:, :, i] .= tmp
		end
		# OrthonormalBasis = tmp

		# display(OrthonormalBasis)
		ExpansionCoefficients = zeros(T, NumberOfTerms)

		return new{T}(
			InputDistribution,
			input_dimensions,
			ExpansionDegree,
			NumberOfTerms,
			MultivariatePolynomialDegrees,
			OrthonormalRepresentation,
			OrthonormalBasis,
			ExpansionCoefficients,
		)
	end
end



function aPC_MultivariatePolynomialDegrees_old(N, d)
	# Input:
	# N- Number of uncertain parameters
	# d - Degree of polynomial expansion
	# Output:
	# PolynomialDegree - Multivariate Polynomial Degrees 
	# Total number of terms
	P = numberPolynomials(N, d)
	# Possible Degrees
	UniqueDegreeCombinations = zeros((d + 1)^N, d)
	PossibleDegrees = [collect(0:d) for _ in N:-1:1]
	if N == 1
		UniqueDegreeCombinations = PossibleDegrees[1]
	else
		tmp = cat(collect(ndgrid_array(reverse(PossibleDegrees)...))...; dims = 3)
		UniqueDegreeCombinations = reshape(tmp, :, N)
	end
	# Possible degree computation
	DegreeWeight = zeros(1, size(UniqueDegreeCombinations, 1))
	for i ∈ 1:1:size(UniqueDegreeCombinations, 1)
		DegreeWeight[i] = 0.0
		for j ∈ 1:1:N
			DegreeWeight[i] = DegreeWeight[i] + UniqueDegreeCombinations[i, j]
		end
	end
	# Sorting of possible degree
	id = sortperm(DegreeWeight; dims = 2)[:]
	SortDegreeCombinations = UniqueDegreeCombinations[id, :]
	# Multivariate Polynomial Degrees  
	reverse_columns!(SortDegreeCombinations)
	return SortDegreeCombinations[1:P, :]
end


function sort_basis_indices(keys; graded = false, reverse = false)
	if reverse
		reverse!(keys, dims = 1)
	end
	indices = sortperm(keys[:, 1])
	if graded
		sums = sum(keys[indices, :], dims = 2)
		graded_indices = sortperm(sums, dims = 1)
		indices = indices[graded_indices]
	end
	return indices
end


function aPC_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T; use_p = false, p::Float64 = 0.85) where {T <: Integer}
	# Initialize the indices for the first parameter
	range_ = 0:max_degree |> collect
	indices = reshape(range_, :, 1)  # Make it a column vector

	for di in 1:num_dimensions-1
		indices = repeat(indices, inner = (max_degree + 1, 1))
		front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di)
		indices = hcat(front, indices)
		if use_p
			# Apply truncation using p-norm sparsity
			idx_to_keep = vec(sum((indices ./ (num_dimensions + 1)) .^ p, dims = 2) .^ (1 / p) .<= 1)
			indices = indices[idx_to_keep, :]
		else
			indices = indices[vec(sum(indices; dims = 2)).<=max_degree, :]
		end
	end

	indices = hcat(vec(sum(indices; dims = 2)), indices)
	indices = sortslices(indices; dims = 1, rev = false)[:, 2:end]
	reverse_columns!(indices)
	return indices
end



function aPC_PsiPolynomialMatrix(apc::aPC{T}, TrainingInput) where {T <: Real}
	NumberOfTerms, InputDimensions = size(apc.MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = ones(T, NumberOfTerms, NCpoints)
	for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		for j ∈ 1:NCpoints  # For each input sample
			product = 1.0  # Initialize the product for this term and sample
			for ii ∈ 1:InputDimensions  # For each dimension of the input
				degree = apc.MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
				coeffs = apc.OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
				p = Polynomials.Polynomial(coeffs)  # Create the polynomial
				x = reduce(hcat, TrainingInput)[ii, j]
				# @show degree coeffs p x p(x) 

				product *= p(x)  # Evaluate the polynomial at x and multiply
			end
			Psi[i, j] = product  # Assign the product to Psi matrix
		end
	end

	return Psi
end




function mean(x)
	s = zero(eltype(x))
	for i in x
		s += i
	end
	return s / length(x)
end

function var(x::AbstractArray)
	m = mean(x)
	s = zero(eltype(x))
	for i in x
		s += (i - m)^2
	end
	return s / (length(x) - 1)
end

function aPC_OrthonormalBasis(Data, Degree)
	d = Degree #Degree of polinomial expansion
	dd = d + 1 #Degree of polinomial for roots defenition
	L_norm = 1 # L-norm for polnomial normalization
	NumberOfDataPoints = length(Data)
	# @info "Construction of Arbitrary Polynomial Basis" --> We do d+2 x d+2, to have one order higher polynomials to get the gaussian quadrature poitns in the end!!!

	MeanOfData = mean(Data)
	VarOfData = var(Data)
	Data = Data ./ MeanOfData
	m = zeros(2 * dd + 2)
	for i ∈ 0:(2*dd+1)
		m[i+1] = sum(Data .^ i) / NumberOfDataPoints # Raw Moments
	end
	poly = zeros(dd + 1, dd + 1)
	MHankel = zeros(dd + 1, dd + 1) # Allocate once for all :)
	for degree ∈ 0:dd
		Hankel = @views MHankel[1:degree+1, 1:degree+1]
		Hankel .= 0.0
		fr = copy(Hankel)
		# fr1 = copy(fr)
		Vc = zeros(degree + 1)
		PolyCoeff_NonNorm = copy(Hankel)

		for i ∈ 0:degree
			for j ∈ 0:degree
				if i < degree
					Hankel[i+1, j+1] = m[i+j+1] # put in the moment
				elseif (i == degree) && (j < degree)
					Hankel[i+1, j+1] = 0.0
				elseif (i == degree) && (j == degree)
					Hankel[i+1, j+1] = 1.0
				end
			end
			# fr1 = copy(Hankel); # Control Hankel only considering the raw moments without division by max(abs) in each row
			Hankel[i+1, :] = Hankel[i+1, :] / maximum(abs.(Hankel[i+1, :]))
		end

		for i ∈ 0:degree
			if (i < degree)
				Vc[i+1] = 0
			elseif (i == degree)
				Vc[i+1] = 1
			end
		end

		# inv_Mm = pinv(Hankel, atol=1e-6)
		# display(Hankel)
		# dt = copy(Vc)
		# fr .= Hankel 
		Vp = Hankel \ Vc
		PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Vp
		if (100 * abs(sum(abs.(Hankel * PolyCoeff_NonNorm[degree+1, 1:degree+1])) - sum(abs.(Vc))) > 0.5)
			@warn "Computational error of the linear solver is too high"
		end


		#Normalization of polynomial coefficients
		P_norm = 0
		for i ∈ 1:NumberOfDataPoints
			Poly = 0
			for k ∈ 0:degree
				Poly += PolyCoeff_NonNorm[degree+1, k+1] * Data[i]^k
			end
			P_norm += Poly^2 / NumberOfDataPoints
		end

		for k ∈ 0:degree
			poly[degree+1, k+1] = PolyCoeff_NonNorm[degree+1, k+1] / sqrt(P_norm)
		end


	end

	# Backward linear transformation to the data space
	Data = Data * MeanOfData
	for k ∈ 1:lastindex(poly, 2)
		poly[:, k] = poly[:, k] ./ (MeanOfData^(k - 1))
	end

	#%% Data-driven Arbitrary Orthonormal Polynomial Basis
	OrthonormalBasis = poly
	return OrthonormalBasis # SMatrix{dd+1,dd+1}(
end

function KMeansCollocation(apc::aPC{T}, M = 10)::RowVecs{T} where {T <: Real}
	alg = InducingPoints.KmeansAlg(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat,Z) |> Array |> transpose |> RowVecs
	return Z
end

function kDPPCollocation(apc::aPC{T}, M = 10)::RowVecs{T} where {T <: Real}
	kernel = SqExponentialKernel()
	alg = kDPP(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)';kernel)
	Z = reduce(hcat,Z) |> Array |> transpose |> RowVecs
	return Z
end

function RandomSubsetCollocation(apc::aPC{T}, M = 10)::RowVecs{T} where {T <: Real}
	alg = RandomSubset(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat,Z) |> Array |> transpose |> RowVecs
	return Z
end

function CoverTreeCollocation(apc::aPC{T}, c = 0.2)::RowVecs{T} where {T <: Real}
	alg = CoverTree(c)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat,Z) |> Array |> transpose |> RowVecs
	return Z
end

function UniGridCollocation(apc::aPC{T}, M = 10)::RowVecs{T} where {T <: Real}
	alg = UniGrid(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat,Z) |> Array |> transpose |> RowVecs
	return Z
end


function GaussianCollocation(apc::aPC{T}; strategy = :PCM)::RowVecs{T} where {T <: Real}
	polynomial_roots = zeros(T, apc.input_dimensions, apc.ExpansionDegree + 1)
	for d ∈ 1:apc.input_dimensions
		polynomial_basis = apc.OrthonormalBasis[:, :, d]
		polynomial_roots[d, :] = T.(Polynomials.roots(Polynomials.Polynomial(polynomial_basis[apc.ExpansionDegree+2, :])))
	end
	# polynomial_roots = Real.(reverse(polynomial_roots))
	display(polynomial_roots)
	PointsVector = 1:apc.ExpansionDegree+1 |> collect
	UniqueCombinations = stack(reduce(vcat, collect(Iterators.product([PointsVector for i in 1:apc.input_dimensions]...))))' |> collect
	# DigitalPointsWeight = zeros(size(UniqueCombinations, 1))
	# for (i, r) in enumerate(eachcol(UniqueCombinations))
	# 	DigitalPointsWeight[i] = sum(r)
	# end
	# index_SDPW = sortperm(reshape(DigitalPointsWeight, (:, 1)); dims = 1)[:]
	# SortUniqueCombinations = UniqueCombinations[index_SDPW, :]
	# display(UniqueCombinations)
	# @info size(UniqueCombinations)
	sort_indices = sortperm(sum(UniqueCombinations; dims=2);dims=1)
	# display(sort_indices)
	SortUniqueCombinations = UniqueCombinations[sort_indices[:], :]
	# display(SortUniqueCombinations)
	if strategy == :FT
		TrainingInput = SortUniqueCombinations
		return  RowVecs(Array{T}(view(Float64.(TrainingInput), :, (1:size(TrainingInput, 2)))))
	elseif strategy == :PCM
		temp = abs.(polynomial_roots .-  StatsBase.mean(apc.InputDistribution; dims = 1)[:,:][1])
		temp_sort = mapslices(sortperm, temp, dims=2)
		sorted_polynomial_roots = copy(polynomial_roots)
		for i in axes(sorted_polynomial_roots, 1)
			sorted_polynomial_roots[i, :] = sorted_polynomial_roots[i, temp_sort[i, :]]
		end
		collocation_points = zeros(T, (apc.NumberOfTerms, apc.input_dimensions))
		# display(sorted_polynomial_roots)
		# display(SortUniqueCombinations)
		for i in 1:apc.NumberOfTerms
			for j in axes(SortUniqueCombinations, 2)
				collocation_points[i, j] = sorted_polynomial_roots[j, Int(SortUniqueCombinations[i, j]) ]
			end
		end
		
		collocation_points = sortslices(collocation_points, dims=1, by=x->x[1])
		return RowVecs(Array{T}(view(Float64.(collocation_points), :, (1:size(collocation_points, 2)))))
	end

end


function GaussianCollocation2(apc::aPC{T}) where {T <: Real}

	py"""
	import numpy as np
	import math
	def compute_collocation_points(data, orthonormal_basis):
		# Number of uncertain parameters
		N = data.shape[1]
		# Degree of polynomial expansion
		d = int(orthonormal_basis.shape[0]-2)
		# Number of terms in polynomial expansion
		P = math.factorial(N + d) / (math.factorial(N) * math.factorial(d))
	
		# Compute the roots of the polynomial of degree d+1 for each of the orthogonal basis (each uncertainty parameter)
		polynomial_roots = np.zeros((N, d + 1))
		for i in range(N):
			polynomial_coefficient = orthonormal_basis[d+1, :, i]
			polynomial_roots[i, :] = np.roots(np.flip(polynomial_coefficient))
	
		# Creation of all the possible combinations of the different polynomial_roots for the N uncertainty parameters
		nroot_para_mat = np.tile(np.arange(1, d + 2), (N, 1))  # Matrix with the number of roots for each parameter
		unique_combinations = np.array(np.meshgrid(*(row for row in nroot_para_mat))).T.reshape(-1, N)
	
		# Sort the unique_combinations based on the sum of each row. Later this is going to be the ranking to construct the
		# most probable collocation points using the most probable polynomial roots.
		sort_unique_combinations = unique_combinations[np.argsort(unique_combinations.sum(axis=1)), :].astype(float)
	
		# Sort the polynomial_roots based on the higher probability of occurrence. In this case, we assume that the higher
		# probability of occurrence is the mean value, and sort the values with respect con the distance from the mean
		temp = abs(np.subtract(polynomial_roots, np.transpose(np.mean(data, axis=0, keepdims=True))))
		temp_sort = np.argsort(temp, axis=1)
		sorted_polynomial_roots = polynomial_roots.copy()
		for i, row in enumerate(sorted_polynomial_roots):
			sorted_polynomial_roots[i, :] = row[temp_sort[i]]
	
		# print(sorted_polynomial_roots)
		# print(sort_unique_combinations)
		# Compute the most probable collocation points as the combination of the most probable polynomial roots for each
		# parameter.
		for i in range(0, sort_unique_combinations.shape[0]):
			for j in range(0, sort_unique_combinations.shape[1]):
				sort_unique_combinations[i, j] = sorted_polynomial_roots[j, int(sort_unique_combinations[i, j]) - 1]
	
		# Choose the P most probable collocation points (being P the order of expansion)
		collocation_points = sort_unique_combinations[0:int(P), :]
		x=1
	
		return collocation_points
		"""

	data = reduce(hcat, apc.InputDistribution) |> transpose
	orthonormal_basis = apc.OrthonormalBasis |> Array
	Training =  py"compute_collocation_points($data, $orthonormal_basis)" |> Array{T} 
	collocation_points = sortslices(Training, dims=1, by=x->x[1])

	return collocation_points |> RowVecs
end


function numberPolynomials(n::Int64, d::Int64)
	x, y = max(d, n), min(d, n)
	return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end


function reverse_columns!(x)
	for row in axes(x, 1)
		x[row, :] = reverse(x[row, :])
	end
end

function train!(apc::aPC{T}, TrainingInput::RowVecs, TrainingOutput::RowVecs) where {T <: Real}
	@info "=> aPC Toolbox: Training Arbitrary Polynomial Chaos ..."
	Psi = aPC_PsiPolynomialMatrix(apc, TrainingInput)'
	to = reduce(vcat, TrainingOutput)
	# Psi_inv = pinv(Psi)
	# apc.ExpansionCoefficients = Psi_inv * to
	x₀ = apc.ExpansionCoefficients
	apc.ExpansionCoefficients = invert(Matrix(Psi), to, Lₖx₀(2, x₀); alg = :gcv_svd, method = LBFGS())

	return nothing
end


function predict(apc::aPC{T}, PredictionInput) where {T <: Real}
	@info "=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ..."
	Psi = aPC_PsiPolynomialMatrix(apc, PredictionInput)'
	PredictionOutput = [dot(apc.ExpansionCoefficients, row) for row in eachrow(Psi)]
	return PredictionOutput
end


function UQ(apc::aPC{T}) where {T <: Float64}
	@info "=> aPC Toolbox: UQ Arbitrary Polynomial Chaos ..."
	lc = Array{T}(apc.ExpansionCoefficients)
	OutputMean = Vector{Float64}(lc[1, :])
	OutputVar = Vector{Float64}(sum(lc[2:end, :] .^ 2; dims = 1)[:])
	return (OutputMean = OutputMean, OutputVar = OutputVar)
end
