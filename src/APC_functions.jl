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

import Optim: NewtonTrustRegion, Options, optimize, minimizer, minimum, LBFGS, IPNewton
import RegularizationTools: Lₖx₀, solve, RegularizationProblem, setupRegularizationProblem, to_general_form, to_standard_form, gcv_tr, gcv_svd, invert, Lₖ, NelderMead, LₖB, Lₖx₀B, LₖDₓ, Lₖx₀Dₓ, LₖDₓB, Lₖx₀DₓB
using Combinatorics
using CUDA
using Enzyme
using InducingPoints
using LazyArrays
using LazyGrids
using LinearAlgebra
using LineSearches
using MKL
using Polyester
using PolynomialRoots
using Polynomials
using PrettyTables
using SparseArrays
using StatsBase
using Strided
using TensorOperations
using UnicodePlots # To use spy from SparseArrays
using Zygote

Strided.set_num_threads(Threads.nthreads())
LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

mutable struct aPC{T <: Float64}
	const InputDistribution::RowVecs{T} # in [ d x N-samples]
	const input_dimensions::Int64
	output_dimensions::Int64
	const ExpansionDegree::Int64
	const NumberOfTerms::Int64
	const MultivariatePolynomialDegrees::AbstractArray{Int64}
	const OrthonormalRepresentation::Bool
	const OrthonormalBasis::Array{T}
	ExpansionCoefficients::Matrix{T}

	# Constructor
	function aPC(
		InputDistribution::RowVecs{T},
		ExpansionDegree::Int64;
		OrthonormalRepresentation::Bool = true,
		qnorm::Float64 = 1.0,
		outdim::Int64 = 1,
	) where T
		input_dimensions = Int64(size(InputDistribution[1], 1))
		MultivariatePolynomialDegrees = aPC_MultivariatePolynomialDegrees(input_dimensions, ExpansionDegree; qnorm = qnorm)
		NumberOfTerms = min(size(MultivariatePolynomialDegrees, 1), numberPolynomials(ExpansionDegree, input_dimensions))
		if qnorm != 1.0
			@info "qnorm reduced the number of terms from $(numberPolynomials(ExpansionDegree, input_dimensions)) to $NumberOfTerms"
		end
		OrthonormalBasis = zeros(ExpansionDegree + 2, ExpansionDegree + 2, input_dimensions)
		@inbounds for i in 1:input_dimensions
			tmp = aPC_OrthonormalBasis(getindex.(InputDistribution, i), ExpansionDegree)
			OrthonormalBasis[:, :, i] .= tmp
		end
		ExpansionCoefficients = zeros(T, NumberOfTerms, outdim)

		return new{T}(
			InputDistribution,
			input_dimensions,
			outdim,
			ExpansionDegree,
			NumberOfTerms,
			MultivariatePolynomialDegrees,
			OrthonormalRepresentation,
			OrthonormalBasis,
			ExpansionCoefficients,
		)
	end
end

import Base.show

function show(io::IO, apc::aPC)
	println(io, "=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ...")
	println(io, "aPC{$(typeof(apc).parameters[1])} Summary:")
	println(io, "Input Dimensions: ", apc.input_dimensions)
	println(io, "Output Dimensions: ", apc.output_dimensions)
	println(io, "Expansion Degree: ", apc.ExpansionDegree)
	println(io, "Number Of Terms: ", apc.NumberOfTerms)
	println(io, "Orthonormal Representation: ", apc.OrthonormalRepresentation ? "Yes" : "No")
	# Depending on the size, you might want to only show a preview of the arrays
	println(io, "Multivariate Polynomial Degrees: ", size(apc.MultivariatePolynomialDegrees))
	println(io, "Orthonormal Basis: Dimensions ", size(apc.OrthonormalBasis))
	println(io, "Expansion Coefficients: Length ", length(apc.ExpansionCoefficients))
end

# function aPC_MultivariatePolynomialDegrees_old(N, d)
# 	# Input:
# 	# N- Number of uncertain parameters
# 	# d - Degree of polynomial expansion
# 	# Output:
# 	# PolynomialDegree - Multivariate Polynomial Degrees 
# 	# Total number of terms
# 	P = numberPolynomials(N, d)
# 	# Possible Degrees
# 	UniqueDegreeCombinations = zeros((d + 1)^N, d)
# 	PossibleDegrees = [collect(0:d) for _ in N:-1:1]
# 	if N == 1
# 		UniqueDegreeCombinations = PossibleDegrees[1]
# 	else
# 		tmp = cat(collect(ndgrid_array(reverse(PossibleDegrees)...))...; dims = 3)
# 		UniqueDegreeCombinations = reshape(tmp, :, N)
# 	end
# 	# Possible degree computation
# 	DegreeWeight = zeros(1, size(UniqueDegreeCombinations, 1))
# 	for i ∈ 1:1:size(UniqueDegreeCombinations, 1)
# 		DegreeWeight[i] = 0.0
# 		for j ∈ 1:1:N
# 			DegreeWeight[i] = DegreeWeight[i] + UniqueDegreeCombinations[i, j]
# 		end
# 	end
# 	# Sorting of possible degree
# 	id = sortperm(DegreeWeight; dims = 2)[:]
# 	SortDegreeCombinations = UniqueDegreeCombinations[id, :]
# 	# Multivariate Polynomial Degrees  
# 	reverse_columns!(SortDegreeCombinations)
# 	return SortDegreeCombinations[1:P, :]
# end


# function sort_basis_indices(keys; graded = false, reverse = false)
# 	if reverse
# 		reverse!(keys, dims = 1)
# 	end
# 	indices = sortperm(keys[:, 1])
# 	if graded
# 		sums = sum(keys[indices, :], dims = 2)
# 		graded_indices = sortperm(sums, dims = 1)
# 		indices = indices[graded_indices]
# 	end
# 	return indices
# end

function normalization_functions(matrix)
	# Calculate mean and std for each column
	col_means = StatsBase.mean(matrix, dims = 1)
	col_stds = StatsBase.std(matrix, dims = 1)
	# Define the normalization function
	normalize = (x) -> (x .- col_means) ./ col_stds
	# Define the inverse normalization function
	inverse_normalize = (x) -> x .* col_stds .+ col_means
	return normalize, inverse_normalize
end


function aPC_MultivariatePolynomialDegrees(num_dimensions, max_degree; qnorm = 1.0)
	# Initialize the indices for the first parameter
	range_ = 0:max_degree |> collect
	indices = reshape(range_, :, 1)  # Make it a column vector
	@inbounds for di in 1:num_dimensions-1
		indices = repeat(indices, inner = (max_degree + 1, 1))
		front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di)
		indices = ApplyArray(hcat, front, indices)
		if qnorm != 1.0
			idx_to_keep = vec(sum((indices ./ (max_degree + 1)) .^ qnorm, dims = 2) .^ (1.0 / qnorm) .<= 1.0)
			indices = indices[idx_to_keep, :]
			# @info "Q-norm removed $(length(idx_to_keep)-sum(idx_to_keep)) terms"
		else
			indices = indices[vec(sum(indices; dims = 2)).<=max_degree, :]
		end
	end
	indices .= sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
	reverse_columns!(indices)
	return indices
end

function aPC_PsiPolynomialMatrix(apc::aPC{T}, TrainingInput) where {T <: Real}
	NumberOfTerms, InputDimensions = size(apc.MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = ones(NumberOfTerms, NCpoints)
	TrainingInput = reduce(hcat, TrainingInput)
	@inbounds for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		@batch for j ∈ 1:NCpoints  # For each input sample
			product = 1.0  # Initialize the product for this term and sample
			for ii ∈ 1:InputDimensions  # For each dimension of the input
				degree = @views apc.MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
				coeffs = @views apc.OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
				x = @views TrainingInput[ii, j]
				p = Polynomials.Polynomial{T}(coeffs)  # Create the polynomial
				product *= evalpoly(x, p)  # Evaluate the polynomial at x and multiply
			end
			Psi[i, j] = product  # Assign the product to Psi matrix
		end
	end
	return Psi
end

function mean(x)
	s = zero(eltype(x))
	@simd for i in x
		s += i
	end
	return s / length(x)
end

function var(x)
	m = mean(x)
	s = zero(eltype(x))
	@simd for i in x
		s += (i - m)^2
	end
	return s / (length(x) - 1)
end

@inbounds function aPC_OrthonormalBasis(Data, Degree)
	d = Degree #Degree of polinomial expansion
	dd = d + 1 #Degree of polinomial for roots defenition
	# L_norm = 1 # L-norm for polnomial normalization
	NumberOfDataPoints = length(Data)
	# @info "Construction of Arbitrary Polynomial Basis" --> We do d+2 x d+2, to have one order higher polynomials to get the gaussian quadrature poitns in the end!!!
	MeanOfData = mean(Data)
	# VarOfData = var(Data)
	Data = Data ./ MeanOfData
	m = zeros(2 * dd + 2)
	@simd for i ∈ 0:(2*dd+1)
		m[i+1] = sum(Data .^ i) / NumberOfDataPoints # Raw Moments
	end
	poly = zeros(dd + 1, dd + 1)
	MHankel = zeros(dd + 1, dd + 1) # Allocate once for all :)
	@inbounds for degree ∈ 0:dd
		Hankel = @views MHankel[1:degree+1, 1:degree+1]
		Vc = zeros(degree + 1)
		PolyCoeff_NonNorm = copy(Hankel)
		for i ∈ 0:degree
			@batch for j ∈ 0:degree
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

		@simd for i ∈ 0:degree
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
		Vp = similar(Vc)
		try
			Vp .= Hankel \ Vc
		catch
			@warn "Hankel matrix singular, trying pseudo inverse."
			Vp .= pinv(Hankel) * Vc
		end
		# Vp = Hankel \ Vc
		PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Vp
		if (100 * abs(sum(abs.(Hankel * PolyCoeff_NonNorm[degree+1, 1:degree+1])) - sum(abs.(Vc))) > 0.5)
			@warn "Computational error of the linear solver is too high"
		end

		#Normalization of polynomial coefficients
		P_norm = 0
		for i ∈ 1:NumberOfDataPoints
			Poly = 0
			@simd for k ∈ 0:degree
				Poly += @views PolyCoeff_NonNorm[degree+1, k+1] * Data[i]^k
			end
			P_norm += Poly^2 / NumberOfDataPoints
		end
		@simd for k ∈ 0:degree
			poly[degree+1, k+1] = @views PolyCoeff_NonNorm[degree+1, k+1] / sqrt(P_norm)
		end
	end

	# Backward linear transformation to the data space
	Data = Data * MeanOfData
	@inbounds for k ∈ 1:lastindex(poly, 2)
		poly[:, k] = @views poly[:, k] ./ (MeanOfData^(k - 1))
	end

	#%% Data-driven Arbitrary Orthonormal Polynomial Basis
	return poly
end

function KMeansCollocation(apc, M = 10)
	alg = InducingPoints.KmeansAlg(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function kDPPCollocation(apc, M = 10)
	kernel = SqExponentialKernel()
	alg = kDPP(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)'; kernel)
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function RandomSubsetCollocation(apc, M = 10)
	alg = RandomSubset(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function CoverTreeCollocation(apc, c = 0.2)
	alg = CoverTree(c)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end

function UniGridCollocation(apc, M = 10)
	alg = UniGrid(M)
	Z = inducingpoints(alg, reduce(hcat, apc.InputDistribution)')
	Z = reduce(hcat, Z) |> Array |> transpose |> RowVecs
	return Z
end


function GaussianCollocation(apc::aPC{T}; strategy = :PCM) where {T <: Real}
	@info apc
	polynomial_roots = zeros(apc.input_dimensions, apc.ExpansionDegree + 1)
	@inbounds for d ∈ Base.oneto(Int64(apc.input_dimensions))
		polynomial_basis = @views apc.OrthonormalBasis[:, :, d]
		polynomial_roots[d, :] = @view reinterpret(T, PolynomialRoots.roots(@views polynomial_basis[apc.ExpansionDegree+2, :]))[1:2:end-1]
	end
	PointsVector = 1:apc.ExpansionDegree+1 |> collect
	UniqueCombinations = stack(reduce(vcat, (Iterators.product([PointsVector for _ in 1:apc.input_dimensions]...))))'


	sort_indices = sortperm(sum(UniqueCombinations; dims = 2); dims = 1)
	SortUniqueCombinations = UniqueCombinations[sort_indices[:], :]
	if strategy == :FT
		TrainingInput = SortUniqueCombinations
		return RowVecs(view(TrainingInput, :, (1:size(TrainingInput, 2))))
	elseif strategy == :PCM
		temp = abs.(polynomial_roots .- StatsBase.mean(apc.InputDistribution; dims = 1)[:, :][1])
		temp_sort = mapslices(sortperm, temp, dims = 2)
		@inbounds for i in axes(polynomial_roots, 1)
			polynomial_roots[i, :] = @views polynomial_roots[i, temp_sort[i, :]]
		end
		collocation_points = zeros(apc.NumberOfTerms, apc.input_dimensions)
		@inbounds for i in 1:apc.NumberOfTerms
			for j in axes(SortUniqueCombinations, 2)
				collocation_points[i, j] = @views polynomial_roots[j, Int(SortUniqueCombinations[i, j])]
			end
		end
		collocation_points = @strided sortslices(collocation_points, dims = 1, by = x -> x[1])
		return RowVecs(view(collocation_points, :, (1:size(collocation_points, 2))))
	end
end

function numberPolynomials(n, d)
	x, y = max(d, n), min(d, n)
	return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end


function reverse_columns!(x)
	@inbounds for row in axes(x, 1)
		x[row, :] = reverse(@views x[row, :])
	end
end

function train!(apc::aPC{T}, TrainingInput, TrainingOutput; bayesian_inversion = :true,reg_mode = 3) where {T <: Real}
	@info "=> aPC Toolbox: Training Arbitrary Polynomial Chaos ..."
	@info apc
	y_rhs = reduce(hcat, TrainingOutput)'
	if apc.output_dimensions != size(y_rhs, 2)
		# @warn "Output dimensions of the aPC model and the training output do not match"
		apc.output_dimensions = size(y_rhs, 2)
		apc.ExpansionCoefficients = zeros(T, apc.NumberOfTerms, apc.output_dimensions)
	end
	# NumberOfTerms, InputDimensions = size(apc.MultivariatePolynomialDegrees)
	# NCpoints = size(TrainingInput, 1)
	# Psi = SMatrix{NumberOfTerms,NCpoints}(aPC_PsiPolynomialMatrix(apc, TrainingInput)')
	Psi = Matrix{T}(aPC_PsiPolynomialMatrix(apc, TrainingInput)')
	# @warn "SPYING"
	# display(UnicodePlots.spy(sparse(Psi)))
	# @debug "" size(TrainingInput) size(TrainingOutput) size(Psi) typeof(Psi) typeof(TrainingOutput) typeof(TrainingInput) size(apc.ExpansionCoefficients) typeof(apc.ExpansionCoefficients)
	# Psi_inv = pinv(Psi;rtol= sqrt(eps(real(float(oneunit(eltype(Psi)))))) )
	Psi_inv = pinv(Psi; rtol = sqrt(eps(real(float(oneunit(eltype(Psi)))))))
	# Psi_inv = pinv(Psi;rtol=0.6)
	# @debug "" size(y_rhs) typeof(y_rhs)  typeof(apc.ExpansionCoefficients) size(Psi_inv) size(Psi)
	@tensor opt = true apc.ExpansionCoefficients[i, k] = Psi_inv[i, j] * y_rhs[j, k]
	# apc.ExpansionCoefficients = Psi_inv * y_rhs
	# @info "" size(C) typeof(C)
	# apc.ExpansionCoefficients .= C
	# @debug "" Psi_inv * y_rhs typeof(Psi_inv * y_rhs)
	# @einsum ExpansionCoefficients[i,k] := Psi_inv[i,j] * y_rhs[i,k]
	# apc.ExpansionCoefficients .= ExpansionCoefficients
	if bayesian_inversion
		@info "Using bayesian regularization y_rhs find the expansion coefficients"
		x₀ = apc.ExpansionCoefficients # Quite a good first guess :) And pinv is quite stable.
		# apc.ExpansionCoefficients .= reshape(reduce(hcat,[invert(Psi, y_rhs[:, i], Lₖx₀(2, @view x₀[:,i]);  alg = :gcv_svd, method = LBFGS(linesearch=LineSearches.BackTracking())) for i in axes(y_rhs, 2)]), :, apc.output_dimensions)
		# apc.ExpansionCoefficients .= reshape(reduce(hcat,[invert(Psi, y_rhs[:, i], Lₖx₀(0, @view x₀[:,i]);  alg = :gcv_svd, method = LBFGS(linesearch=LineSearches.BackTracking())) for i in axes(y_rhs, 2)]), :, apc.output_dimensions)
		# apc.ExpansionCoefficients .= reshape(reduce(hcat,[solve(setupRegularizationProblem(Psi,y_rhs[:,i],@view x₀[:,i])) for i in axes(y_rhs, 2)]), :, apc.output_dimensions)
		# x̂ = deepcopy(abs.(apc.ExpansionCoefficients .= reshape(reduce(hcat,[solve(setupRegularizationProblem(Psi,y_rhs[:,i],@view x₀[:,i])) for i in axes(y_rhs, 2)]), :, apc.output_dimensions)
		# ))
		@batch for i in axes(y_rhs, 2)
			# lower = zeros(size(Psi, 2)) .+ 0.001
			# upper = ones(size(lower)) .+ 80
			# x₀[x₀[:, i].<0.0, i] .= 0.1
			# apc.ExpansionCoefficients[:, i] .= invert(Psi'*Psi .+ 1.0*Diagonal(ones(size(Psi,2))), Psi'*y_rhs[:, i], Lₖx₀(3, view(x₀,:, i));alg = :gcv_svd, method = LBFGS(linesearch = LineSearches.BackTracking()))
			apc.ExpansionCoefficients[:, i] .= invert(Psi, y_rhs[:, i], Lₖx₀(reg_mode, view(x₀, :, i)); alg = :gcv_svd, method = LBFGS(linesearch = LineSearches.BackTracking()))

		end
	end
	for k in axes(apc.ExpansionCoefficients, 2)
		res = (@views sqrt(mean((Psi * apc.ExpansionCoefficients[:, k] .- y_rhs[:, k]) .^ 2)))
		@info "Error for axis $k" res
	end
	# @warn "SPYING"
	# display(UnicodePlots.spy(sparse(apc.ExpansionCoefficients)))
	# @info "" sqrt(mean((Psi * apc.ExpansionCoefficients .- y_rhs) .^ 2))
	return nothing
end

# function predict(apc::aPC{T}, PredictionInput) where {T <: Real}
# 	@info "=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ..."
# 	Psi = aPC_PsiPolynomialMatrix(apc, PredictionInput)
# 	PredictionOutput = zeros(size(Psi, 2))
# 	@batch for i ∈ axes(Psi, 2)
# 		PredictionOutput[i] = dot(apc.ExpansionCoefficients, @views Psi[:, i])
# 	end
# 	# PredictionOutput = [dot(apc.ExpansionCoefficients, col) for col in eachcol(Psi)]
# 	return PredictionOutput
# end

function predict(apc::aPC{T}, PredictionInput) where {T <: Real}
	@info "=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ..."
	Psi = aPC_PsiPolynomialMatrix(apc, PredictionInput)
	# @warn "SPYING"
	# display(UnicodePlots.spy(sparse(Psi)))
	# @einsum PredictionOutput[i, j] := Psi[i,k] * apc.ExpansionCoefficients[i, j]
	@tensor opt = true PredictionOutput[k, j] := Psi[i, k] * apc.ExpansionCoefficients[i, j]
	# PredictionOutput = zeros(size(Psi, 2))
	# @batch for i ∈ axes(Psi,2)
	# 	PredictionOutput[i] = dot(apc.ExpansionCoefficients, @views Psi[:, i])
	# end
	return PredictionOutput
end

function UQ(apc::aPC{T}; axis = 1) where {T <: Real}
	# @info "=> aPC Toolbox: UQ Arbitrary Polynomial Chaos ..."
	# @info "Computing the mean and variance of the output for dimension $axis"
	lc = Array{T}(apc.ExpansionCoefficients[:, axis])
	OutputMean = @views lc[1, :]
	OutputVar = @views sum(lc[2:end, :] .^ 2; dims = 1)[:]
	return (OutputMean = OutputMean, OutputVar = OutputVar)
end
