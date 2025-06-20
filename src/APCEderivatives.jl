# Copyright (c) 2024 wildt
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# using Zygote
# using ChainRulesCore
# using ForwardDiff
# using Zygote: @adjoint
# using ChainRules
# using DifferentiationInterface
# using DispatchDoctor: @stable
# using ComponentArrays
# using ReverseDiff
# using Polynomials
# using Tracker

# using Mooncake: @from_rrule, DefaultCtx

function ChainRulesCore.frule((_, Δx), ::typeof(reverse_columns!), x)
    Δx_reversed = similar(Δx)
    for row in axes(Δx, 1)
        Δx_reversed[row, :] = reverse(Δx[row, :])
    end
    y = reverse_columns!(x)
    return y, Δx_reversed
end

# @stable function ∂Ψ(x, degree)
# 	input_dimensions = size(x, 2)
# 	MultivariatePolynomialDegrees = create_Polynomial_Degrees(input_dimensions, degree; qnorm = 1.0)
# 	OrthonormalBasis = create_basis(x, degree; qnorm = 1.0)
# 	return Zygote.jacobian(x -> aPCE_PsiPolynomialMatrix(MultivariatePolynomialDegrees, OrthonormalBasis, x), x) |> first
# end


# @stable function ∂create_basis(x, degree; qnorm = 1.0)
# 	return Zygote.jacobian(x -> create_basis(x, degree; qnorm = qnorm), x) |> first
# end

@stable function ensure_matrix(arr)
    if ndims(arr) == 1
        return reshape(arr, (length(arr), 1))
    else
        return arr
    end
end

# Base.merge(ca::ComponentVector) = ca
# function Base.merge(ca1::ComponentVector{T1}, ca2::ComponentVector{T2}) where {T1,T2}
#     ax = getaxes(ca1)
#     ax2 = getaxes(ca2)
#     vks = valkeys(ax[1])
#     vks2 = valkeys(ax2[1])
#     idxmap = indexmap(ax[1])
#     _p = Vector{promote_type(T1,T2)}()
#     sizehint!(_p, length(ca1)+length(ca2))
#     for vk in vks
#         if vk in vks2
#             _p = vcat(_p, ca2[vk])
#         else
#             _p = vcat(_p, ca1[vk])
#         end
#     end
#     new_idxmap = Vector{Pair{Symbol, Int64}}([])
#     sizehint!(new_idxmap, length(ca2))
#     max_val = maximum(idxmap)
#     for vk in vks2
#         if !(vk in vks)
#             _p = vcat(_p, ca2[vk])
#             new_idxmap = vcat(new_idxmap, [getval(vk)=>max_val+1])
#             max_val += 1
#         end
#     end
#     merged_ax = Axis(merge(idxmap, new_idxmap))
#     ComponentArray(_p, merged_ax)
# end

# Base.merge(ca1::ComponentVector, ca2::ComponentVector, cs::ComponentVector) = merge(merge(ca1,ca2), cs...)

# function Tracker.param(ca::ComponentArray)
#     x = getdata(ca)
#     length(x) == 0 && return ComponentArray(Tracker.param(Float32[]), getaxes(ca))
#     return ComponentArray(Tracker.param(x), getaxes(ca))
# end

# Tracker.extract_grad!(ca::ComponentArray) = Tracker.extract_grad!(getdata(ca))

# function Base.materialize(bc::Base.Broadcast.Broadcasted{Tracker.TrackedStyle, Nothing,
#     typeof(zero), <:Tuple{<:ComponentVector}})
#     ca = first(bc.args)
#     return ComponentArray(zero.(getdata(ca)), getaxes(ca))
# end

# function Base.getindex(g::Tracker.Grads, x::ComponentArray)
#     Tracker.istracked(getdata(x)) || error("Object not tracked: $x")
#     return g[Tracker.tracker(getdata(x))]
# end

# # For TrackedArrays ignore Base.maybeview
# ## Tracker with views doesn't work quite well
# @inline function Base.getproperty(x::ComponentVector{T, <:TrackedArray},
#     s::Symbol) where {T}
#     return getproperty(x, Val(s))
# end

# @inline function Base.getproperty(x::ComponentVector{T, <:TrackedArray}, v::Val) where {T}
#     return ComponentArrays._getindex(Base.getindex, x, v)
# end


# function ChainRulesCore.rrule(::typeof(aPCE_OrthonormalBasis), Data::AbstractArray{T}, Degree::S, normalize_data::Val{false}) where {T<:Real,S<:Integer}
#     NCpoints, inpDim = size(Data)
#     function aPCE_pullback(dy)
#         ∂Data =  ForwardDiff.gradient(x -> sum(aPCE_OrthonormalBasis(x, Degree, Val(false))), Data)
#         ∂∂ = @thunk reduce(hcat, [dy * ∂Data[:, i] for i in 1:inpDim])
#         return ChainRulesCore.NoTangent(), ∂∂
#     end
#     return aPCE_OrthonormalBasis(Data, Degree, Val(false)), aPCE_pullback
# end


# function ChainRulesCore.rrule(::Type{ComponentArray}, nt::NamedTuple)
#     res = ComponentArray(nt)
#     function CA_NT_pullback(Δ::AbstractArray)
#         if length(Δ) == length(res)
#             return (CRC.NoTangent(), NamedTuple(ComponentArray(vec(Δ), getaxes(res))))
#         end
#         error("Got pullback input of shape $(size(Δ)) & type $(typeof(Δ)) for output " *
#               "of shape $(size(res)) & type $(typeof(res))")
#         return nothing
#     end
#     CA_NT_pullback(Δ::ComponentArray) = (@show Δ; (ChainRulesCore.NoTangent(), NamedTuple(Δ)))
#     return res, CA_NT_pullback
# end

# ChainRulesCore.rrule(::Type{ComponentArray}, data, axes) = ComponentArray(data, axes), Δ -> (ChainRulesCore.NoTangent(), getdata(Δ), ChainRulesCore.NoTangent())

# function ChainRulesCore.rrule(::typeof(aPCE_PsiPolynomialMatrix), TrainingInput::AbstractArray{T}, MultivariatePolynomialDegrees, OrthonormalBasis::AbstractArray{S}) where {S, T<:Real}
#     # Forward pass
#     Psi = aPCE_PsiPolynomialMatrix(TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)
# 	x = ensure_matrix(TrainingInput)
# 	Nterms = size(MultivariatePolynomialDegrees, 1)
# 	NCpoints, inpDim = size(x)
# 	function aPCE_PsiPolynomialMatrix_pullback(dy_raw)
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
#         ∂OrthonormalBasis = zeros(S, size(OrthonormalBasis))
#          for degree ∈ 1:size(OrthonormalBasis, 1)
#             for ii ∈ 1:size(OrthonormalBasis, 3)
#                 ∂OrthonormalBasis[degree, :, ii] =  ReverseDiff.gradient(x -> sum(aPCE_OrthonormalBasis(x, degree)), x)
#             end
#         end

#         ∂∂OrthonormalBasis = @thunk reduce(hcat, [dy * ∂OrthonormalBasis[:, i] for i in 1:inpDim])


# 		∂∂ = @thunk reduce(hcat, [dy * ∂x[:, i] for i in 1:inpDim])
# 		return (NoTangent(), ∂∂, NoTangent(), ∂∂OrthonormalBasis, NoTangent())
# 	end
#     return Psi, aPCE_PsiPolynomialMatrix_pullback
# end


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


function ChainRulesCore.rrule(
        ::typeof(compute_Psi_element),
        i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions
    )
    # Forward computation
    product = one(eltype(TrainingInput))
    derivatives = zeros(size(TrainingInput, 2))

    for ii in 1:InputDimensions
        degree = MultivariatePolynomialDegrees[i, ii] + 1
        coeffs = OrthonormalBasis[degree, 1:degree, ii]
        x = TrainingInput[j, ii]
        p_x = evalpoly(x, coeffs)

        # Compute derivative for this dimension
        other_products = prod(
            ii == jj ? evalpoly_derivative(x, coeffs) : evalpoly(x, coeffs)
                for jj in 1:InputDimensions
        )
        derivatives[ii] = other_products

        product *= p_x
    end

    function compute_Psi_element_pullback(dy)
        ∂TrainingInput = @thunk(dy * derivatives)
        return (NoTangent(), NoTangent(), NoTangent(), ∂TrainingInput, NoTangent(), NoTangent(), NoTangent())
    end

    return product, compute_Psi_element_pullback
end


function ChainRulesCore.rrule(
        ::typeof(aPCE_PsiPolynomialMatrix_zygote),
        TrainingInput,
        MultivariatePolynomialDegrees,
        OrthonormalBasis
    )
    # Ensure input is matrix
    x = ensure_matrix(TrainingInput)

    # Forward pass
    Psi = aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, OrthonormalBasis)


    # Extract dimensions
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(x, 1)
    T = eltype(x)

    function aPCE_PsiPolynomialMatrix_pullback(ΔPsi)
        ΔTrainingInput = zeros(T, size(x))

        # Pre-compute polynomial evaluations to avoid redundant calculations
        poly_values = Array{T}(undef, NumberOfTerms, InputDimensions, NCpoints)

        # First pass: compute all polynomial evaluations
        for i in 1:NumberOfTerms
            for d in 1:InputDimensions
                degree = MultivariatePolynomialDegrees[i, d] + 1
                coeffs = @view OrthonormalBasis[degree, 1:degree, d]
                for j in 1:NCpoints
                    x_val = x[j, d]
                    poly_values[i, d, j] = evalpoly_two(x_val, coeffs)
                end
            end
        end

        # Second pass: compute gradients
        for j in 1:NCpoints
            for i in 1:NumberOfTerms
                Δij = ΔPsi[i, j]
                if Δij == zero(T)
                    continue
                end

                for d in 1:InputDimensions
                    degree = MultivariatePolynomialDegrees[i, d]
                    if degree == 0
                        continue  # Derivative of constant is zero
                    end

                    # Compute derivative for dimension d
                    derivative_coeffs = zeros(T, degree)
                    coeffs = @view OrthonormalBasis[degree + 1, 1:(degree + 1), d]
                    for k in 1:degree
                        derivative_coeffs[k] = coeffs[k + 1] * k
                    end

                    x_val = x[j, d]
                    deriv_value = evalpoly_two(x_val, derivative_coeffs)

                    # Compute product of polynomial values for other dimensions
                    other_dims_product = one(T)
                    for other_d in 1:InputDimensions
                        if other_d != d
                            other_dims_product *= poly_values[i, other_d, j]

                            # Early termination if product becomes zero
                            if other_dims_product == zero(T)
                                break
                            end
                        end
                    end

                    # Update gradient
                    ΔTrainingInput[j, d] += Δij * deriv_value * other_dims_product
                end
            end
        end

        return (NoTangent(), ΔTrainingInput, NoTangent(), NoTangent())
    end


    return Psi, aPCE_PsiPolynomialMatrix_pullback
end


function ChainRulesCore.rrule(
        ::typeof(aPCE_PsiPolynomialMatrix),
        TrainingInput,
        MultivariatePolynomialDegrees,
        OrthonormalBasis
    )
    # Ensure input is matrix
    x = ensure_matrix(TrainingInput)

    # Forward pass
    # @info "DEBUGGING"
    # @info MultivariatePolynomialDegrees OrthonormalBasis x
    Psi = aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis)
    # @info "DEBUGGING end "


    # Extract dimensions
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(x, 1)
    T = eltype(x)

    function aPCE_PsiPolynomialMatrix_pullback(ΔPsi)
        ΔTrainingInput = zeros(T, size(x))

        # Pre-compute polynomial evaluations to avoid redundant calculations
        poly_values = Array{T}(undef, NumberOfTerms, InputDimensions, NCpoints)

        # First pass: compute all polynomial evaluations
        for i in 1:NumberOfTerms
            for d in 1:InputDimensions
                degree = MultivariatePolynomialDegrees[i, d] + 1
                coeffs = @view OrthonormalBasis[degree, 1:degree, d]
                for j in 1:NCpoints
                    x_val = x[j, d]
                    poly_values[i, d, j] = evalpoly_two(x_val, coeffs)
                end
            end
        end

        # Second pass: compute gradients
        for j in 1:NCpoints
            for i in 1:NumberOfTerms
                Δij = ΔPsi[i, j]
                if Δij == zero(T)
                    continue
                end

                @batch for d in 1:InputDimensions
                    degree = MultivariatePolynomialDegrees[i, d]
                    if degree == 0
                        continue  # Derivative of constant is zero
                    end

                    # Compute derivative for dimension d
                    derivative_coeffs = zeros(T, degree)
                    coeffs = @view OrthonormalBasis[degree + 1, 1:(degree + 1), d]
                    for k in 1:degree
                        derivative_coeffs[k] = coeffs[k + 1] * k
                    end

                    x_val = x[j, d]
                    deriv_value = evalpoly_two(x_val, derivative_coeffs)

                    # Compute product of polynomial values for other dimensions
                    other_dims_product = one(T)
                    for other_d in 1:InputDimensions
                        if other_d != d
                            other_dims_product *= poly_values[i, other_d, j]

                            # Early termination if product becomes zero
                            if other_dims_product == zero(T)
                                break
                            end
                        end
                    end

                    # Update gradient
                    ΔTrainingInput[j, d] += Δij * deriv_value * other_dims_product
                end
            end
        end

        return (NoTangent(), ΔTrainingInput, NoTangent(), NoTangent())
    end

    return Psi, aPCE_PsiPolynomialMatrix_pullback
end


### TensorOperations Tricks!


using ChainRulesCore
using ForwardDiff

function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray{T}, d_expansion::Int; center_data = false) where {T}
    # Forward pass
    basis = create_basis(x, d_expansion; center_data = center_data)

    # Define the pullback
    function create_basis_pullback(Δbasis)
        # Use ForwardDiff to compute the gradient
        function basis_wrapper(x_vec)
            x_reshaped = reshape(x_vec, size(x))
            basis = create_basis(x_reshaped, d_expansion; center_data = center_data)
            # Return a scalar value for gradient computation
            return sum(basis .* Δbasis)
        end

        # Compute gradient using ForwardDiff
        grad = ForwardDiff.gradient(basis_wrapper, vec(x))

        # Reshape gradient to match input shape
        grad_reshaped = reshape(grad, size(x))

        # Return the gradient with respect to x
        return (NoTangent(), grad_reshaped, NoTangent())
    end

    return basis, create_basis_pullback
end

# Mooncake.jl version of the rrule for create_basis
# @from_rrule DefaultCtx Tuple{
#     typeof(create_basis),
#     AbstractArray, Integer,
# }

# Enzyme rules
# using Enzyme

# # Enzyme rule for reverse_columns!
# function Enzyme.autodiff(::Enzyme.ReverseMode, ::typeof(reverse_columns!), x::AbstractArray)
#     x_reversed = similar(x)
#     for row in axes(x, 1)
#         x_reversed[row, :] = reverse(x[row, :])
#     end
#     return x_reversed
# end

# # Enzyme rule for compute_Psi_element
# function Enzyme.autodiff(::Enzyme.ReverseMode, ::typeof(compute_Psi_element), i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions)
#     # Forward computation
#     product = one(eltype(TrainingInput))
#     derivatives = zeros(size(TrainingInput, 2))

#     for ii in 1:InputDimensions
#         degree = MultivariatePolynomialDegrees[i, ii] + 1
#         coeffs = OrthonormalBasis[degree, 1:degree, ii]
#         x = TrainingInput[j, ii]
#         p_x = evalpoly(x, coeffs)

#         # Compute derivative for this dimension
#         other_products = prod(
#             ii == jj ? evalpoly_derivative(x, coeffs) : evalpoly(x, coeffs)
#                 for jj in 1:InputDimensions
#         )
#         derivatives[ii] = other_products

#         product *= p_x
#     end

#     return product, derivatives
# end

# # Enzyme rule for aPCE_PsiPolynomialMatrix_zygote
# function Enzyme.autodiff(::Enzyme.ReverseMode, ::typeof(aPCE_PsiPolynomialMatrix_zygote), TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)
#     # Ensure input is matrix
#     x = ensure_matrix(TrainingInput)

#     # Forward pass
#     Psi = aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, OrthonormalBasis)

#     # Extract dimensions
#     NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
#     NCpoints = size(x, 1)
#     T = eltype(x)

#     # Pre-compute polynomial evaluations to avoid redundant calculations
#     poly_values = Array{T}(undef, NumberOfTerms, InputDimensions, NCpoints)

#     # First pass: compute all polynomial evaluations
#     for i in 1:NumberOfTerms
#         for d in 1:InputDimensions
#             degree = MultivariatePolynomialDegrees[i, d] + 1
#             coeffs = @view OrthonormalBasis[degree, 1:degree, d]
#             for j in 1:NCpoints
#                 x_val = x[j, d]
#                 poly_values[i, d, j] = evalpoly_two(x_val, coeffs)
#             end
#         end
#     end

#     return Psi, poly_values
# end

# # Enzyme rule for create_basis
# function Enzyme.autodiff(::Enzyme.ReverseMode, ::typeof(create_basis), x::AbstractArray{T}, d_expansion::Int; center_data = false) where {T}
#     # Forward pass
#     basis = create_basis(x, d_expansion; center_data = center_data)

#     # Compute gradient using Enzyme's autodiff
#     function basis_wrapper(x_vec)
#         x_reshaped = reshape(x_vec, size(x))
#         basis = create_basis(x_reshaped, d_expansion; center_data = center_data)
#         return sum(basis)
#     end

#     # Return both the basis and a function to compute gradients
#     return basis, basis_wrapper
# end
