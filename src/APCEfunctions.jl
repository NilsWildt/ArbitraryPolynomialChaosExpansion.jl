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
using Combinatorics
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


export create_basis!, evaluate_Ψ_zygote
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


@stable function aPCE_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T; qnorm = 1.0)::Matrix{T} where {T<:Integer}
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
	indices = sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
	reverse_columns!(indices)
	return indices
end



# 	function compute_sample_polynomial_product(sample::Vector{T}, term_degrees::Vector{Int}, OrthonormalBasis) where T <: Real
#     product = 1.0
#     for ii in eachindex(term_degrees)
#         degree = term_degrees[ii] + 1
#         coeffs = OrthonormalBasis[degree, 1:degree, ii]
#         x = sample[ii]
#         product *= evalpoly_two(x, coeffs)
#     end
#     return product
# end


# function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis) where {T <: Real}
#     NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
#     NCpoints = size(TrainingInput, 1)
#     # Initialize Psi matrix without filling it in a mutating loop
#     Psi = [compute_sample_polynomial_product(TrainingInput[j, :], MultivariatePolynomialDegrees[i, :], OrthonormalBasis) for i in 1:NumberOfTerms, j in 1:NCpoints]
#     return Psi
# end

@stable function aPCE_PsiPolynomialMatrix_zygote(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis) where {T <: Real}
	NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = Zygote.Buffer(ones(eltype(TrainingInput), NumberOfTerms, NCpoints))
	# OrthonormalBasis = T.(OrthonormalBasis)
	# Function to evaluate polynomials for a given term and input sample
	for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		# product = 1.0  # Initialize the product for this term and sample
		for ii ∈ 1:InputDimensions  # For each dimension of the input
			degree = MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
			coeffs = OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
			p = Polynomials.Polynomial{T}(coeffs)  # Create the polynomial
			# p = Poly(coeffs)
			for j ∈ 1:NCpoints  # For each input sample
				x = TrainingInput[j, ii]
				Psi[i, j] *= p(x)  # Evaluate the polynomial at x and multiply
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
			@batch for j ∈ 1:NCpoints  # For each input sample
				x = @views TrainingInput[j, ii]
				Psi[i, j] *= p(x)  # Evaluate the polynomial at x and multiply
				# Psi[i,j] *= evalpoly(x, p)
			end
		end
	end
	return Psi
end

@stable function aPCE_PsiPolynomialMatrix(TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{S}) where {S,T <: Real}
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
			@batch for j ∈ 1:NCpoints  # For each input sample
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
			@batch for j ∈ 1:NCpoints  # For each input sample
				x = @views TrainingInput[j, ii]
				Psi[i, j] *= p(x)  # Evaluate the polynomial at x and multiply
				# Psi[i,j] *= evalpoly(x, p)
			end
		end
	end
	return Psi
end


@stable function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S; normalize_data = false) where {T<:Real,S<:Integer}
	ChainRulesCore.ignore_derivatives() do 
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
		m = zeros(T, 2 * dd + 2)
		for l ∈ 0:(2*dd+1)
			m[l+1] = sum(Data .^ l) / NumberOfDataPoints
		end
		# if any(isnan, m)
		# 	@warn "NaNs in the moments"
		# 	@info "" Data NumberOfDataPoints 
		# end
		OrthonormalBasis = zeros(T, dd + 1, dd + 1)
		OrthogonalBasis = zeros(T, dd + 1, dd + 1) # Allocate once for all :)
		@inbounds for degree ∈ 0:dd
			Hankel = @views OrthogonalBasis[1:degree+1, 1:degree+1]
			Vc = zeros(T, degree + 1)
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
			if (deviation > 0.5)
				@warn "Computational error of the linear solver is too high: $(round(deviation;digits=3))"
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
		return OrthonormalBasis
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

@stable function create_basis(x, degree;normalize_data=true)
	@ignore_derivatives begin
		input_dimensions = size(x, 2)
		OrthonormalBasis = zeros(eltype(x), degree + 1, degree + 1, input_dimensions)
		@inbounds for i in 1:input_dimensions
			tmp = aPCE_OrthonormalBasis(x[:, i], degree;normalize_data=normalize_data)
			OrthonormalBasis[:, :, i] = tmp
		end
		return OrthonormalBasis
	end
end

@stable function create_basis!(OrthonormalBasis, x, degree;normalize_data=false)
	@ignore_derivatives begin
		input_dimensions = size(x, 2)
		# OrthonormalBasis = eltype(x).(OrthonormalBasis)
		if eltype(OrthonormalBasis) != eltype(x)
			OrthonormalBasis = eltype(x).(OrthonormalBasis)
		end
		# OrthonormalBasis = zeros(eltype(x), degree + 1, degree + 1, input_dimensions)
		@inbounds for i in 1:input_dimensions
			tmp = aPCE_OrthonormalBasis(x[:, i], degree;normalize_data=normalize_data)
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
@stable function evaluate_Ψ(x, coeffs, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name) 
	T = eltype(coeffs)
	Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@tensoropt PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
	# @info "" typeof(x) typeof(coeffs) typeof(MultivariatePolynomialDegrees) typeof(OrthonormalBasis) typeof(degree) typeof(name)
	return T.(PredictionOutput)
end

@stable function evaluate_Ψ(x, coeffs::ReverseDiff.TrackedArray, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name)
	Ψ = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	@einsum PredictionOutput[k, j] := Ψ[k, i] * coeffs[i, j]
	return PredictionOutput
end

@stable function evaluate_Ψ(x, coeffs::AbstractVecOrMat{T}, MultivariatePolynomialDegrees, OrthonormalBasis, degree, name) where {T<:ForwardDiff.Dual}
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

@stable function train(Ψ::AbstractArray{T}, y_rhs; bayesian_inversion = :true, reg_mode = 3) where {T <: Real}
	NumberOfTerms = size(Ψ, 2)
	output_dimensions = size(y_rhs, 2)
	coeffs = zeros(T, NumberOfTerms, output_dimensions)
	# Ψ = Matrix{T}(Ψ)
	# display(UnicodePlots.spy(sparse(Ψ)))
	Psi_inv = pinv(Ψ; rtol = sqrt(eps(real(float(oneunit(eltype(Ψ)))))))

	@einsum coeffs[i, k] = Psi_inv[i, j] * y_rhs[j, k] # tensoropt
	if bayesian_inversion
		@info "Using bayesian regularization y_rhs find the expansion coefficients"
		x₀ = coeffs # Quite a good first guess :) And pinv is 
		for i in axes(y_rhs, 2) # stride=true 
			@info "Bayesian regularization for axis $i"
			coeffs[:, i] .= invert(Ψ, y_rhs[:, i], Lₖx₀(reg_mode, view(x₀, :, i)); alg = :gcv_svd, method = LBFGS(linesearch = LineSearches.BackTracking()))
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
