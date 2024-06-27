# Copyright (c) 2024 wildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
using Zygote
using ChainRulesCore
using ForwardDiff
using Zygote: @adjoint
using ChainRules
# using DifferentiationInterface
using DispatchDoctor: @stable
using ComponentArrays

function ChainRulesCore.frule((_, Δx), ::typeof(reverse_columns!), x)
	Δx_reversed = similar(Δx)
	for row in axes(Δx, 1)
		Δx_reversed[row, :] = reverse(Δx[row, :])
	end
	y = reverse_columns!(x)
	return y, Δx_reversed
end

@stable function ChainRulesCore.rrule(::typeof(reverse_columns!), x)
	function reverse_columns_pullback(Δy)
		Δx = similar(Δy)
		for row in axes(Δy, 1)
			Δx[row, :] = reverse(Δy[row, :])
		end
		return (NO_FIELDS, Δx)
	end
	y = reverse_columns!(x)
	return y, reverse_columns_pullback
end

@stable function ∂Ψ(x, degree)
	input_dimensions = size(x, 2)
	MultivariatePolynomialDegrees = create_Polynomial_Degrees(input_dimensions, degree; qnorm = 1.0)
	OrthonormalBasis = create_basis(x, degree; qnorm = 1.0)
	return Zygote.jacobian(x -> aPCE_PsiPolynomialMatrix(MultivariatePolynomialDegrees, OrthonormalBasis, x), x) |> first
end
@stable function ∂create_basis(x, degree; qnorm = 1.0)
	return Zygote.jacobian(x -> create_basis(x, degree; qnorm = qnorm), x) |> first
end

@stable function ensure_matrix(arr)
	if ndims(arr) == 1
		return reshape(arr, (length(arr), 1))
	else
		return arr
	end
end

# @stable function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	# @info "" typeof(x) typeof(MultivariatePolynomialDegrees) typeof(OrthonormalBasis) typeof(degree)
# 	x = ensure_matrix(x)
# 	Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	Nterms = size(MultivariatePolynomialDegrees, 1)
# 	NCpoints, inpDim = size(x)
# 	function compose_Ψ_pullback(dy_raw)
# 		dy = unthunk(dy_raw)
# 		∂x = ones(Nterms, inpDim)
# 		@inbounds for i in 1:Nterms
# 			for j in 1:NCpoints
# 				for ii in 1:inpDim
# 					# oldval = ∂x[i, ii]
# 					∂x[i, ii] = 1.0
# 					@batch for kk in 1:inpDim # @batch
# 						degree_k = MultivariatePolynomialDegrees[i, kk] + 1
# 						coeffs = @views OrthonormalBasis[degree_k, 1:degree_k, kk]
# 						# pp = Polynomials.Polynomial(coeffs)
# 						if ii == kk
# 							# n = length(coeffs) - 1
# 							# p = Poly([i * coeffs[i+1] for i in 1:n])
# 							∂x[i, ii] *= evaluate_derivative_horner(x[j, ii], coeffs)
# 							# ∂x[i, ii] *= p(x[j, ii])
# 						else
# 							# p = Poly(coeffs)
# 							# ∂x[i, ii] *= p(x[j, ii])
# 							∂x[i, ii] *= evalpoly_two(x[j, ii], coeffs)
# 						end
# 					end
# 				end
# 			end
# 		end
# 		∂∂ = @thunk reduce(hcat, [dy * ∂x[:, i] for i in 1:inpDim])
# 		return (NoTangent(), ∂∂, NoTangent(), @thunk(dy * Ψforward'), NoTangent())
# 	end
# 	return Ψforward, compose_Ψ_pullback
# end

# @stable function ChainRulesCore.rrule(::typeof(compose_Ψ), x::ComponentArrays.ComponentVector, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	# @info "" typeof(x) typeof(MultivariatePolynomialDegrees) typeof(OrthonormalBasis) typeof(degree)
# 	OrthonormalBasis = create_basis(x.value, Layer.d_expansion; normalize_data = Layer.normalize_data) #
# 	x = ensure_matrix(x.value)
# 	Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	Nterms = size(MultivariatePolynomialDegrees, 1)
# 	NCpoints, inpDim = size(x)
# 	function compose_Ψ_pullback(dy_raw)
# 		dy = unthunk(dy_raw)
# 		∂x = ones(Nterms, inpDim)
# 		@inbounds for i in 1:Nterms
# 			for j in 1:NCpoints
# 				for ii in 1:inpDim
# 					# oldval = ∂x[i, ii]
# 					∂x[i, ii] = 1.0
# 					@batch for kk in 1:inpDim # @batch
# 						degree_k = MultivariatePolynomialDegrees[i, kk] + 1
# 						coeffs = @views OrthonormalBasis[degree_k, 1:degree_k, kk]
# 						# pp = Polynomials.Polynomial(coeffs)
# 						if ii == kk
# 							# n = length(coeffs) - 1
# 							# p = Poly([i * coeffs[i+1] for i in 1:n])
# 							∂x[i, ii] *= evaluate_derivative_horner(x[j, ii], coeffs)
# 							# ∂x[i, ii] *= p(x[j, ii])
# 						else
# 							# p = Poly(coeffs)
# 							# ∂x[i, ii] *= p(x[j, ii])
# 							∂x[i, ii] *= evalpoly_two(x[j, ii], coeffs)
# 						end
# 					end
# 				end
# 			end
# 		end
# 		∂∂ = @thunk reduce(hcat, [dy * ∂x[:, i] for i in 1:inpDim])
# 		return (NoTangent(), ∂∂, NoTangent(), ComponentVector(prior = zeros(size(x)), value = @thunk(dy * Ψforward')), NoTangent())
# 	end
# 	return Ψforward, compose_Ψ_pullback
# end
