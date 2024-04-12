# Copyright (c) 2024 wildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
using Zygote
using ChainRulesCore
using ForwardDiff
using Zygote: @adjoint


function ChainRulesCore.frule((_, Δx), ::typeof(reverse_columns!), x)
	Δx_reversed = similar(Δx)
	for row in axes(Δx, 1)
		Δx_reversed[row, :] = reverse(Δx[row, :])
	end
	y = reverse_columns!(x)
	return y, Δx_reversed
end

function ChainRulesCore.rrule(::typeof(reverse_columns!), x)
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

function ∂Ψ(x, degree)
	input_dimensions = size(x, 2)
	MultivariatePolynomialDegrees = create_Polynomial_Degrees(input_dimensions, degree; qnorm = 1.0)
	OrthonormalBasis = create_basis(x, degree; qnorm = 1.0)
	return Zygote.jacobian(x -> aPCE_PsiPolynomialMatrix(MultivariatePolynomialDegrees, OrthonormalBasis, x), x) |> first
end
function ∂create_basis(x, degree; qnorm = 1.0)
	return Zygote.jacobian(x -> create_basis(x, degree; qnorm = qnorm), x) |> first
end

function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	OrthonormalBasis = create_basis(x, degree)
	Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	Nterms = size(MultivariatePolynomialDegrees, 1)
	NCpoints, inpDim = size(x)
	# project_x = ProjectTo(x)
	function compose_Ψ_pullback(dy_raw)
		dy = unthunk(dy_raw)
		∂x = ones(NCpoints, Nterms * NCpoints)
		for j in 1:NCpoints
			for i in 1:Nterms
				for ii in 1:inpDim
					all_prod = 1.0  # Product of all polynomial evaluations except dimension ii
					for kk in 2:inpDim
						degree = MultivariatePolynomialDegrees[i, kk] + 1
						coeffs = @views OrthonormalBasis[degree, 1:degree, kk]
						# all_prod *= evalpoly_two(x[j, ii], coeffs)
						all_prod *= evalpoly(x[j, kk], coeffs)
					end
					# Derivative for dimension ii
					# dc = derivative_coeffs(OrthonormalBasis[MultivariatePolynomialDegrees[i, ii] + 1, :, ii])
					degree = MultivariatePolynomialDegrees[i, ii] + 1
					coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
					derivative_eval = evalpoly(x[j, ii], Polynomials.derivative(Polynomial(coeffs)))
					# Accumulate gradient
					∂x[j, i*j] *= derivative_eval * all_prod / evalpoly(x[j, ii], Polynomial(coeffs))
				end
			end
			# @info "" size(∂x[j, :]) size(dy')
			# ∂x[j, :] = sum(dy'.*reshape(∂x[j,:],size(dy));dims=1)
		end
		# @info "" mean(dy)
		# (∇,) = AD.jacobian(AD.ForwardDiffBackend(), x ->  compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree), x	)
		∂∂ = sum(dy' * ∂x; dims = 2)
		# @info "" size(∂∂)
		return (ChainRules.NoTangent(), ∂∂, ChainRules.NoTangent(), ChainRules.NoTangent(), ChainRules.NoTangent())
	end
	# @warn "Something in this derivative is still wrong, use ForwardDiff for now"
	return Ψforward, compose_Ψ_pullback
end