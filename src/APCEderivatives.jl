# Copyright (c) 2024 wildt
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
using Zygote
using ChainRulesCore
using ForwardDiff
using Zygote: @adjoint
using ChainRules
using DifferentiationInterface

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

function ensure_matrix(arr)
	if ndims(arr) == 1
		return reshape(arr, (length(arr), 1))
	else
		return arr
	end
end



# function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	OrthonormalBasis = create_basis(x, degree)
# 	Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	backend = DifferentiationInterface.AutoSparseForwardDiff()
# 	project_x = ProjectTo(x)
# 	function compose_Ψ_pullback(dy_raw)
# 		# @info "" size(dy_raw)
# 		dy =  unthunk(dy_raw)
# 		∂x = DifferentiationInterface.jacobian(x ->  compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree), backend,x) 
# 		# display()

# 		# ∂x = Zygote.jacobian(x ->  compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree), x) |> first
# 		# ∂∂ = dy' * ∂x
# 		@ignore_derivatives begin
# 			@info ""  size(dy') size(∂x) size(x) 
# 			# @info repeat(∂∂,1,NCpoints)'
# 			# @info size(repeat(∂∂,1,length(x)))
# 			@info size(dy_raw)
# 			@info size(∂x)
# 			@info project_x(dy) 
# 			# @info project_x(∂∂)
# 			@info size(project_x(x))
# 		∂∂ = dy' * ∂x
# 		@info "" size(∂∂) size(dy') size(∂x) size(x)
# 			# @info dy dy_raw
# 		end

# 		return (ChainRules.NoTangent(), ∂x, ChainRules.NoTangent(), ChainRules.NoTangent(), ChainRules.NoTangent())
# 	end
# 	# @warn "Something in this derivative is still wrong, use ForwardDiff for now"
# 	return Ψforward, compose_Ψ_pullback
# end


# function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	OrthonormalBasis = create_basis(x, degree)
# 	Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	Nterms = size(MultivariatePolynomialDegrees, 1)	
# 	x = ensure_matrix(x)
# 	NCpoints, inpDim = size(x)
# 	project_x = ProjectTo(x)
# 	n,m = size(x)
# 	@info "" n m
# 	function compose_Ψ_pullback(dy_raw)
# 		dy =  unthunk(dy_raw)
# 		∂x = ones(NCpoints, Nterms * NCpoints)
# 		for j in 1:NCpoints
# 			@batch for i in 1:Nterms
# 				for ii in 1:inpDim
# 					all_prod = 1.0  # Product of all polynomial evaluations except dimension ii
# 					for kk in 2:inpDim
# 						degree = MultivariatePolynomialDegrees[i, kk] + 1
# 						coeffs = @views OrthonormalBasis[degree, 1:degree, kk]
# 						# all_prod *= evalpoly_two(x[j, ii], coeffs)
# 						all_prod *= evalpoly(x[j, kk], coeffs)
# 					end
# 					# Derivative for dimension ii
# 					# dc = derivative_coeffs(OrthonormalBasis[evaluate_Ψ[i, ii] + 1, :, ii])
# 					degree = MultivariatePolynomialDegrees[i, ii] + 1
# 					coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
# 					derivative_eval = evalpoly(x[j, ii], Polynomials.derivative(Polynomial(coeffs)))
# 					# Accumulate gradient
# 					∂x[j, j] *= derivative_eval * all_prod / evalpoly(x[j, ii], Polynomial(coeffs))
# 				end
# 			end
# 		end
# 		# @info "" mean(dy)
# 		# (∇,) = AD.jacobian(AD.ForwardDiffBackend(), x ->  compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree), x	)
# 		# ∂∂ = @thunk sum(@thunk dy'*∂x; dims = 2)
# 		∂∂ = dy' * ∂x
# 		# Ψforward = fill(1000.0,NCpoints, Nterms)
# 		# ∂∂ = fill(1000.0,n,Nterms)
# 		@ignore_derivatives begin
# 			@info "" size(∂∂) size(dy') size(∂x) size(x) size(Ψforward)
# 			# @info repeat(∂∂,1,NCpoints)'
# 			# @info project_x(dy) 
# 			# @info project_x(∂∂)
# 			# @info size(project_x(x)) 
# 			# @info size(repeat(∂∂,1,length(x)))
# 			display(dy)
# 			# @info dy dy_raw
# 		end
# 		return (ChainRules.NoTangent(), ∂∂, ChainRules.NoTangent(), Ψforward, ChainRules.NoTangent())
# 	end
# 	# @warn "Something in this derivative is still wrong, use ForwardDiff for now"
# 	return Ψforward, compose_Ψ_pullback
# end


function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	# Input validation
	# if any(isnan, x) || any(isinf, x) ||
	#    any(isnan, MultivariatePolynomialDegrees) || any(isinf, MultivariatePolynomialDegrees) ||
	#    any(isnan, OrthonormalBasis) || any(isinf, OrthonormalBasis)
	# 	error("Input contains NaNs or Infs")
	# end

	Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
	Nterms = size(MultivariatePolynomialDegrees, 1)
	NCpoints, inpDim = size(x)
	project_x = ProjectTo(x)

	function compose_Ψ_pullback(dy_raw)
		dy = unthunk(dy_raw)
		∂x = ones(Nterms, inpDim)

		@inbounds for i in 1:Nterms
			for j in 1:NCpoints
				for ii in 1:inpDim
					oldval = ∂x[i, ii]
					∂x[i, ii] = 1.0
					@batch for kk in 1:inpDim
						degree_k = MultivariatePolynomialDegrees[i, kk] + 1
						coeffs = @views OrthonormalBasis[degree_k, 1:degree_k, kk]
						if ii == kk
							∂x[i, ii] *= evaluate_derivative_horner(x[j, ii], coeffs)
							# ∂x[i, ii] *= evaluate_derivative_horner(x[j, ii], coeffs)
						else
							# t1 = @elapsed Poly(coeffs)(x[j, ii])
							# t2 = @elapsed evalpoly_two(x[j, ii], coeffs)
							# @info "t1: $t1, t2: $t2"
							∂x[i, ii] *= evalpoly_two(x[j, ii], coeffs)
						end
						# if isnan(∂x[i, ii]) || isinf(∂x[i, ii])
						#     @warn "Detected NaN or Inf in ∂x" i j kk ii oldval ∂x[i, ii] coeffs x[j, ii]
						#     @infiltrate
						# end
					end
					# ∂x[:, ii] = dy * ∂x[:, ii]
				end
			end
		end

		# if any(isnan, ∂x) || any(isinf, ∂x)
		#     @warn "NaNs or Infs detected in ∂x"
		#     display(∂x)
		# end

		∂∂ = reduce(hcat, [dy * ∂x[:, i] for i in 1:inpDim])

		# if any(isnan, ∂∂) || any(isinf, ∂∂) || any(isnan, dy) || any(isinf, dy) || any(isnan, Ψforward) || any(isinf, Ψforward)
		#     @warn "Detected NaNs or Infs"
		#     display(Dict("∂∂" => ∂∂, "dy" => dy, "Ψforward" => Ψforward, "∂x" => ∂x))
		# end
		# @info "" size(Ψforward) size(dy)
		return (NoTangent(), ∂∂, NoTangent(), dy * Ψforward', NoTangent())
		# return (NoTangent(), ∂x, NoTangent(), Ψforward, NoTangent())

	end

	return Ψforward, compose_Ψ_pullback
end


# function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
# 	Nterms, inpDim = size(MultivariatePolynomialDegrees) # P: Number of terms, N: Number of dimensions
# 	N_data = size(x, 1) # Total number of points/samples
# 	@inbounds function compose_Ψ_pullback(ȳ)
# 		∂x = ones(N_data, inpDim)
# 		for i in 1:Nterms # For each term
# 			for j in 1:N_data # For each data point
# 				for ii in 1:inpDim # For each dimension
# 					degree = MultivariatePolynomialDegrees[i, ii] + 1
# 					coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
# 					xc = @views x[j, ii]
# 					for kk in 1:inpDim # Dimensions to take derivatives in :) 
# 						if kk != ii
# 							# ∂x[j, ii] *=  evalpoly(xc, Polynomial(coeffs)) * ȳ[j, i]
# 							∂x[j, ii] *= evalpoly_two(xc, coeffs)
# 						else
# 							# ∂x[i,j] *=  evalpoly(xc, derivative(Polynomial(coeffs)))
# 							∂x[j, ii] *= evalpoly_two(xc, derivative_coeffs(coeffs))
# 						end
# 					end
# 				end
# 			end
# 		end
# 		display(∂x)
# 		@tensor test[i,j] := ∂x[k,j] * ȳ[k,i]
# 		@show size(∂x) size(ȳ) size(test)
# 		display(test)
# 		return (NoTangent(), ∂x, NoTangent(), Ψforward, NoTangent())
# 	end
# 	return Ψforward, compose_Ψ_pullback
# end

# function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
#     # Create the orthonormal basis
#     OrthonormalBasis = create_basis(x, degree)
#     # Compute the forward pass of compose_Ψ
#     Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)

#     # Ensure x is a matrix and get its size
#     x = ensure_matrix(x)
#     NCpoints, inpDim = size(x)

#     # Function to project to the same space as x
#     project_x = ProjectTo(x)

#     function compose_Ψ_pullback(dy_raw)
#         dy = unthunk(dy_raw)  # Unthunk the gradient if necessary
#         ∂x = zeros(NCpoints, inpDim)  # Initialize the gradient with respect to x

#         @info "Starting gradient accumulation..."
#         @info "NCpoints: $NCpoints, inpDim: $inpDim, size(dy): $(size(dy))"

#         # Loop through each data point
#         for j in 1:NCpoints
#             # @info "Processing data point $j..."
#             # Loop through each input dimension
#             for ii in 1:inpDim
#                 total_derivative = 0.0
#                 # @info "Processing input dimension $ii..."
#                 # Loop through each component of dy (output dimensions)
#                 for k in 1:size(dy, 2)
#                     # Accumulate the product of polynomials excluding the ii-th dimension
#                     all_prod = 1.0
#                     for kk in 1:inpDim
#                         if kk != ii
#                             degree = MultivariatePolynomialDegrees[k, kk] + 1
#                             coeffs = @views OrthonormalBasis[degree, 1:degree, kk]
#                             eval_res = evalpoly(x[j, kk], coeffs)
#                             all_prod *= eval_res
#                             # @info "all_prod updated: $all_prod (eval_res: $eval_res, x[$j, $kk]: $(x[j, kk]))"
#                         end
#                     end

#                     # Compute the derivative for the ii-th dimension
#                     degree = MultivariatePolynomialDegrees[k, ii] + 1
#                     coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
#                     derivative_eval = evalpoly(x[j, ii], Polynomials.derivative(Polynomial(coeffs)))
#                     eval_original = evalpoly(x[j, ii], Polynomial(coeffs))
#                     # @info "derivative_eval: $derivative_eval (eval_original: $eval_original, x[$j, $ii]: $(x[j, ii]))"

#                     # Accumulate the gradient
#                     gradient_contribution = dy[j, k] * derivative_eval * all_prod / eval_original
#                     total_derivative += gradient_contribution
#                     # @info "Gradient contribution: $gradient_contribution, total_derivative: $total_derivative"
#                 end
#                 ∂x[j, ii] = total_derivative
#                 # @info "∂x[$j, $ii] set to $total_derivative"
#             end
#         end

#         # Since ∂x is already in the same shape as x, we do not need to project again
#         ∂∂ = project_x(∂x)

#         # Debugging information (optional, can be removed in production code)
#         @ignore_derivatives begin
#             @info "" size(∂∂) size(dy) size(∂x) size(x)
#             # display(dy)
#         end

#         # Return the pullback values
#         return (ChainRules.NoTangent(), ∂x, ChainRules.NoTangent(), ChainRules.NoTangent(), ChainRules.NoTangent())
#     end

#     # Return the forward pass value and the pullback function
#     return Ψforward, compose_Ψ_pullback
# end

# function ChainRulesCore.rrule(::typeof(compose_Ψ), x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)
#     # Create the orthonormal basis
#     OrthonormalBasis = create_basis(x, degree)
#     # Compute the forward pass of compose_Ψ
#     Ψforward = compose_Ψ(x, MultivariatePolynomialDegrees, OrthonormalBasis, degree)

#     # Determine the number of terms in the polynomial and the dimensions of the input
#     Nterms = size(MultivariatePolynomialDegrees, 1)
#     x = ensure_matrix(x)
#     NCpoints, inpDim = size(x)
#     project_x = ProjectTo(x)

#     function compose_Ψ_pullback(dy_raw)
#         dy = unthunk(dy_raw)  # Unthunk the gradient if necessary
#         ∂x = zeros(NCpoints, inpDim)  # Initialize the gradient with respect to x

#         # Loop through each data point
#         @inbounds for j in 1:NCpoints
#             # Loop through each polynomial term
#             for i in 1:Nterms
#                 # Loop through each input dimension
#                 for ii in 1:inpDim
#                     all_prod = 1.0  # Initialize the product for dimensions other than ii
#                     for kk in 1:inpDim
#                         if kk != ii
#                             degree =  MultivariatePolynomialDegrees[i, kk] + 1
#                             coeffs = @views OrthonormalBasis[degree, 1:degree, kk]
#                             all_prod *= evalpoly(x[j, kk], coeffs)
# 							# all_prod *= evalpoly_two(x[j, kk], coeffs)
#                         end
#                     end

#                     # Compute the derivative for dimension ii
#                     degree = MultivariatePolynomialDegrees[i, ii] + 1
#                     coeffs = @views OrthonormalBasis[degree, 1:degree, ii]
#                     derivative_eval = evalpoly(x[j, ii], Polynomials.derivative(Polynomial(coeffs)))
#                     # derivative_eval = evalpoly_two(x[j, ii], Polynomials.derivative(Polynomial(coeffs)))
#                     # Accumulate the gradient from each component of dy
# 					# ∂x[j, ii] +=  derivative_eval * all_prod / evalpoly(x[j, ii], Polynomial(coeffs))

#                     # for k in 1:size(dy, 2)
#                         ∂x[j, ii] += dy[j, k] * derivative_eval * all_prod / evalpoly(x[j, ii], Polynomial(coeffs))
#                         # ∂x[j, ii] += sum(dy[j,:]) * derivative_eval * all_prod / evalpoly_two(x[j, ii], Polynomial(coeffs))
#                     # end
#                 end
#             end
#         end

#         # Since ∂x is already in the same shape as x, we do not need to project again
#         # ∂∂ = dy'*∂x |> project_x

#         # Debugging information (optional, can be removed in production code)
#         # @ignore_derivatives begin
#         #     @info "" size(∂∂) size(dy) size(∂x) size(x)
#         #     display(dy)
#         # end

#         # Return the pullback values
#         return (ChainRules.NoTangent(), ∂x, ChainRules.NoTangent(), ChainRules.NoTangent(), ChainRules.NoTangent())
#     end

#     # Return the forward pass value and the pullback function
#     return Ψforward, compose_Ψ_pullback
# end
