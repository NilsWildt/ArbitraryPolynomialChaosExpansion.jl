# Copyright (c) 2024 wildt
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
APCEderivatives.jl

This module provides automatic differentiation support for APCE (Arbitrary Polynomial Chaos Expansion) 
functions using ChainRules.jl. It implements efficient forward and reverse mode differentiation 
rules for polynomial basis construction and evaluation.

Key Features:
- ChainRules support for multiple AD backends (Zygote, ForwardDiff, ReverseDiff, Enzyme, Mooncake)
- Optimized gradient computation for polynomial matrix operations
- Type-stable differentiation rules
- Comprehensive test coverage
"""

using ChainRulesCore
using ForwardDiff
using ReverseDiff
using Mooncake: @from_rrule, DefaultCtx
# using Enzyme
using DispatchDoctor: @stable

# ===== UTILITY FUNCTIONS =====

"""
    ensure_matrix(arr)

Ensure input array is a matrix. If 1D, reshape to column matrix.
"""
@stable function ensure_matrix(arr)
    if ndims(arr) == 1
        return reshape(arr, (length(arr), 1))
    else
        return arr
    end
end


"""
    evalpoly_derivative(x, coeffs)

Helper function to compute polynomial derivative using evalpoly_two.
"""
function evalpoly_derivative(x, coeffs)
    n = length(coeffs) - 1
    derivative_coeffs = coeffs[2:end] .* (1:n)'
    return evalpoly_two(x, derivative_coeffs)
end


# ===== CHAINRULES FOR UTILITY FUNCTIONS =====

"""
    frule((_, Δx), ::typeof(reverse_columns!), x)
ChainRule for reverse_columns! function. Computes gradient by reversing the tangent matrix.
"""
function ChainRulesCore.frule((_, Δx), ::typeof(reverse_columns!), x)
    if isempty(Δx)
        y = reverse_columns!(x)
        return y, Δx
    end

    Δx_reversed = similar(Δx)
    for row in axes(Δx, 1)
        Δx_reversed[row, :] = reverse(Δx[row, :])
    end
    y = reverse_columns!(x)
    return y, Δx_reversed
end

"""
    frule(::RuleConfig, ::typeof(reverse_columns!), x)
More specific method to resolve ambiguity with ChainRulesCore's generic frule fallback.
Returns nothing to indicate no special RuleConfig handling needed.
"""
function ChainRulesCore.frule(::ChainRulesCore.RuleConfig, ::typeof(reverse_columns!), x)
    return nothing
end


# ===== CHAINRULES FOR POLYNOMIAL EVALUATION =====

"""
    rrule(::typeof(compute_Psi_element), i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions)

ChainRule for computing a single element of the Psi matrix. 
Returns the polynomial evaluation and its gradient with respect to the input.
"""
function ChainRulesCore.rrule(
    ::typeof(compute_Psi_element),
    i, j, TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis, InputDimensions
)
    # Forward computation
    product = one(eltype(TrainingInput))
    derivatives = zeros(size(TrainingInput, 2))

    @inbounds for ii in 1:InputDimensions
        degree = MultivariatePolynomialDegrees[i, ii] + 1
        coeffs = @view OrthonormalBasis[degree, 1:degree, ii]
        x = TrainingInput[j, ii]
        p_x = evalpoly_two(x, coeffs)

        # Compute derivative for this dimension using chain rule
        other_products = one(eltype(TrainingInput))
        for jj in 1:InputDimensions
            if ii == jj
                other_products *= evalpoly_derivative(x, coeffs)
            else
                degree_jj = MultivariatePolynomialDegrees[i, jj] + 1
                coeffs_jj = @view OrthonormalBasis[degree_jj, 1:degree_jj, jj]
                other_products *= evalpoly_two(TrainingInput[j, jj], coeffs_jj)
            end
        end
        derivatives[ii] = other_products
        product *= p_x
    end

    function compute_Psi_element_pullback(dy)
        ∂TrainingInput = @thunk(dy * derivatives)
        return (NoTangent(), NoTangent(), NoTangent(), ∂TrainingInput, NoTangent(), NoTangent(), NoTangent())
    end

    return product, compute_Psi_element_pullback
end

# ===== CHAINRULES FOR PSI MATRIX COMPUTATION =====

"""
    rrule(::typeof(aPCE_PsiPolynomialMatrix_zygote), TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)

Efficient ChainRule for the Zygote-compatible Psi matrix computation.
Uses pre-computed polynomial evaluations and optimized gradient computation.
"""
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
        @inbounds for i in 1:NumberOfTerms
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
        @inbounds for j in 1:NCpoints
            for i in 1:NumberOfTerms
                Δij = ΔPsi[i, j]
                iszero(Δij) && continue

                for d in 1:InputDimensions
                    degree = MultivariatePolynomialDegrees[i, d]
                    iszero(degree) && continue  # Derivative of constant is zero

                    # Compute derivative for dimension d
                    derivative_coeffs = zeros(T, degree)
                    coeffs = @view OrthonormalBasis[degree+1, 1:(degree+1), d]
                    for k in 1:degree
                        derivative_coeffs[k] = coeffs[k+1] * k
                    end

                    x_val = x[j, d]
                    deriv_value = evalpoly_two(x_val, derivative_coeffs)

                    # Compute product of polynomial values for other dimensions
                    other_dims_product = one(T)
                    for other_d in 1:InputDimensions
                        if other_d != d
                            other_dims_product *= poly_values[i, other_d, j]
                            # Early termination if product becomes zero
                            iszero(other_dims_product) && break
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

"""
    rrule(::typeof(aPCE_PsiPolynomialMatrix), TrainingInput, MultivariatePolynomialDegrees, OrthonormalBasis)

Efficient ChainRule for the optimized Psi matrix computation.
Uses vectorized operations and pre-computed polynomial evaluations.
"""
function ChainRulesCore.rrule(
    ::typeof(aPCE_PsiPolynomialMatrix),
    TrainingInput,
    MultivariatePolynomialDegrees,
    OrthonormalBasis
)
    # Ensure input is matrix
    x = ensure_matrix(TrainingInput)

    # Forward pass
    Psi = aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis)

    # Extract dimensions
    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(x, 1)
    T = eltype(x)

    function aPCE_PsiPolynomialMatrix_pullback(ΔPsi)
        ΔTrainingInput = zeros(T, size(x))

        # Pre-compute polynomial evaluations to avoid redundant calculations
        poly_values = Array{T}(undef, NumberOfTerms, InputDimensions, NCpoints)

        # First pass: compute all polynomial evaluations
        @inbounds for i in 1:NumberOfTerms
            for d in 1:InputDimensions
                degree = MultivariatePolynomialDegrees[i, d] + 1
                coeffs = @view OrthonormalBasis[degree, 1:degree, d]
                @simd for j in 1:NCpoints
                    x_val = x[j, d]
                    poly_values[i, d, j] = evalpoly_two(x_val, coeffs)
                end
            end
        end

        # Second pass: compute gradients with optimization
        @inbounds for j in 1:NCpoints
            for i in 1:NumberOfTerms
                Δij = ΔPsi[i, j]
                iszero(Δij) && continue

                for d in 1:InputDimensions
                    degree = MultivariatePolynomialDegrees[i, d]
                    iszero(degree) && continue  # Derivative of constant is zero

                    # Compute derivative for dimension d
                    derivative_coeffs = zeros(T, degree)
                    coeffs = @view OrthonormalBasis[degree+1, 1:(degree+1), d]
                    @simd for k in 1:degree
                        derivative_coeffs[k] = coeffs[k+1] * k
                    end

                    x_val = x[j, d]
                    deriv_value = evalpoly_two(x_val, derivative_coeffs)

                    # Compute product of polynomial values for other dimensions
                    other_dims_product = one(T)
                    for other_d in 1:InputDimensions
                        if other_d != d
                            other_dims_product *= poly_values[i, other_d, j]
                            # Early termination if product becomes zero
                            iszero(other_dims_product) && break
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

# ===== CHAINRULES FOR BASIS CREATION =====

"""
    rrule(::typeof(create_basis), x, degree, ::Val{is_ortho}; center_data)

ChainRule for basis creation with Val dispatch.
Uses ForwardDiff for gradient computation of the basis construction.
"""
function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray{T}, degree::Integer, ::Val{is_ortho}; center_data::Bool=true) where {T,is_ortho}
    basis = create_basis(x, degree, Val(is_ortho); center_data=center_data)

    function create_basis_pullback(Δbasis)
        function basis_wrapper(x_vec)
            x_reshaped = reshape(x_vec, size(x))
            basis_val = create_basis(x_reshaped, degree, Val(is_ortho); center_data=center_data)
            return sum(basis_val .* unthunk(Δbasis))
        end
        grad = ForwardDiff.gradient(basis_wrapper, vec(x))
        grad_reshaped = reshape(grad, size(x))
        return (NoTangent(), grad_reshaped, NoTangent(), NoTangent())
    end

    return basis, create_basis_pullback
end

"""
    rrule(::typeof(create_basis), x, degree; center_data)

ChainRule for basis creation with default dispatch (orthonormal basis).
"""
function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray{T}, degree::Integer; center_data::Bool=true) where {T}
    return rrule(create_basis, x, degree, Val(true); center_data=center_data)
end

"""
    rrule(::typeof(create_basis), x, degree, is_orthonormal::Bool; center_data)

ChainRule for basis creation with Bool dispatch.
Converts Bool to Val for type-stable dispatch.
"""
function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray, degree::Integer, is_orthonormal::Bool; center_data::Bool=true)
    return rrule(create_basis, x, degree, Val(is_orthonormal); center_data=center_data)
end


# Enzyme.@import_rrule(typeof(create_basis), AbstractArray, Integer, Val)


ReverseDiff.@grad_from_chainrules create_basis(
    x::ReverseDiff.TrackedArray, d::Integer, is_orthonormal::Val{true}
);
ReverseDiff.@grad_from_chainrules create_basis(
    x::ReverseDiff.TrackedArray, d::Integer, is_orthonormal::Val{false}
);


@from_rrule DefaultCtx Tuple{
    typeof(create_basis),
    AbstractArray,Integer,
}

@from_rrule DefaultCtx Tuple{
    typeof(create_basis),
    AbstractArray,Integer,Val,
}

@from_rrule DefaultCtx Tuple{
    typeof(create_basis),
    AbstractArray,Integer,Bool,
}

# ===== TESTS =====

@testitem "create_basis_chainrules_consistency" begin
    import Pkg
    Pkg.add("Zygote")
    using ForwardDiff, Zygote

    x = rand(20, 2)
    degree = 2

    # Test that all dispatch methods give same gradients
    f1(x) = sum(create_basis(x, degree))
    f2(x) = sum(create_basis(x, degree, Val(true)))
    f3(x) = sum(create_basis(x, degree, true))
    f4(x) = sum(create_basis(x, degree; center_data=true))

    grad1 = ForwardDiff.gradient(f1, x)
    grad2 = ForwardDiff.gradient(f2, x)
    grad3 = ForwardDiff.gradient(f3, x)
    grad4 = ForwardDiff.gradient(f4, x)
    @info "grad1: $grad1"
    @info "grad2: $grad2"
    @info "grad3: $grad3"
    @info "grad4: $grad4"
    @test isapprox(grad1, grad2, atol=1.0e-10)
    @test isapprox(grad1, grad3, atol=1.0e-10)
    @test isapprox(grad1, grad4, atol=1.0e-10)

    # Test Zygote consistency
    grad1_zyg = Zygote.gradient(f1, x)[1]
    @test isapprox(grad1, grad1_zyg, atol=1.0e-8)
end

@testitem "create_basis_differentiation_val_dispatch" begin
    import Pkg
    Pkg.add("Zygote")
    using ForwardDiff, Zygote

    x = rand(15, 2)
    degree = 2

    # Test Val{true} dispatch
    f_true(x) = sum(create_basis(x, degree, Val(true); center_data=true))
    f_true_no_center(x) = sum(create_basis(x, degree, Val(true); center_data=false))

    grad_true_fd = ForwardDiff.gradient(f_true, x)
    grad_true_no_center_fd = ForwardDiff.gradient(f_true_no_center, x)

    grad_true_zyg = Zygote.gradient(f_true, x)[1]
    grad_true_no_center_zyg = Zygote.gradient(f_true_no_center, x)[1]

    @test isapprox(grad_true_fd, grad_true_zyg, atol=1.0e-8)
    @test isapprox(grad_true_no_center_fd, grad_true_no_center_zyg, atol=1.0e-8)

    # Test Val{false} dispatch
    f_false(x) = sum(create_basis(x, degree, Val(false)))

    grad_false_fd = ForwardDiff.gradient(f_false, x)
    grad_false_zyg = Zygote.gradient(f_false, x)[1]

    @test isapprox(grad_false_fd, grad_false_zyg, atol=1.0e-8)
end

@testitem "create_basis_differentiation_bool_dispatch" begin
    import Pkg
    Pkg.add("Zygote")
    using ForwardDiff, Zygote

    x = rand(12, 3)
    degree = 2

    # Test Bool dispatch
    f_bool_true(x) = sum(create_basis(x, degree, true; center_data=false))
    f_bool_false(x) = sum(create_basis(x, degree, false; center_data=true))

    grad_bool_true_fd = ForwardDiff.gradient(f_bool_true, x)
    grad_bool_false_fd = ForwardDiff.gradient(f_bool_false, x)

    grad_bool_true_zyg = Zygote.gradient(f_bool_true, x)[1]
    grad_bool_false_zyg = Zygote.gradient(f_bool_false, x)[1]

    @test isapprox(grad_bool_true_fd, grad_bool_true_zyg, atol=1.0e-8)
    @test isapprox(grad_bool_false_fd, grad_bool_false_zyg, atol=1.0e-8)

    # Test that Bool and Val give same results
    f_val_true(x) = sum(create_basis(x, degree, Val(true); center_data=false))
    f_val_false(x) = sum(create_basis(x, degree, Val(false); center_data=true))

    grad_val_true_fd = ForwardDiff.gradient(f_val_true, x)
    grad_val_false_fd = ForwardDiff.gradient(f_val_false, x)

    @test isapprox(grad_bool_true_fd, grad_val_true_fd, atol=1.0e-12)
    @test isapprox(grad_bool_false_fd, grad_val_false_fd, atol=1.0e-12)
end

@testitem "psi_matrix_chainrules" begin
    import Pkg
    Pkg.add("Zygote")
    using ForwardDiff, Zygote

    # Setup test data
    x = rand(10, 2)
    degree = 2
    MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(2, degree, 1.0, 1.0)
    OrthonormalBasis = create_basis(x, degree, Val(true))

    # Test aPCE_PsiPolynomialMatrix
    f_psi(x) = sum(aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis))

    grad_psi_fd = ForwardDiff.gradient(f_psi, x)
    grad_psi_zyg = Zygote.gradient(f_psi, x)[1]
    @info "compare gradients aPCE_PsiPolynomialMatrix" mean(abs.(grad_psi_fd .- grad_psi_zyg))
    @test isapprox(grad_psi_fd, grad_psi_zyg, atol=1.0e-6)

    # Test aPCE_PsiPolynomialMatrix_zygote
    f_psi_zyg(x) = sum(aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, OrthonormalBasis))

    grad_psi_zyg_fd = ForwardDiff.gradient(f_psi_zyg, x)
    grad_psi_zyg_zyg = Zygote.gradient(f_psi_zyg, x)[1]
    @info "compare gradients aPCE_PsiPolynomialMatrix_zygote" mean(abs.(grad_psi_zyg_fd .- grad_psi_zyg_zyg))

    @test isapprox(grad_psi_zyg_fd, grad_psi_zyg_zyg, atol=1.0e-6)

    # Test that both methods give similar results
    @test isapprox(grad_psi_fd, grad_psi_zyg_fd, atol=1.0e-6)
end


@testitem "reverse_columns_chainrules" begin
    import Pkg
    Pkg.add("Zygote")
    Pkg.add("ChainRulesCore")
    using ChainRulesCore
    using ForwardDiff

    # Test with non-empty matrix
    x = rand(3, 4)
    Δx = rand(3, 4)

    # Test frule manually
    y, Δy = ChainRulesCore.frule((NoTangent(), Δx), reverse_columns!, copy(x))

    # Expected behavior
    x_copy = copy(x)
    reverse_columns!(x_copy)
    expected_y = x_copy
    expected_Δy = similar(Δx)
    for row in axes(Δx, 1)
        expected_Δy[row, :] = reverse(Δx[row, :])
    end

    @test isapprox(y, expected_y)
    @test isapprox(Δy, expected_Δy)

    # Test with empty matrix
    x_empty = Matrix{Float64}(undef, 0, 0)
    Δx_empty = Matrix{Float64}(undef, 0, 0)
    y_empty, Δy_empty = ChainRulesCore.frule((NoTangent(), Δx_empty), reverse_columns!, copy(x_empty))
    @test size(y_empty) == (0, 0)
    @test size(Δy_empty) == (0, 0)
end


@testitem "edge_cases_chainrules" begin
    import Pkg
    Pkg.add("Zygote")
    using ForwardDiff, Zygote

    # Test with minimal data
    x_small = rand(2, 1)
    degree = 1

    f_small(x) = sum(create_basis(x, degree, Val(true)))
    grad_small_fd = ForwardDiff.gradient(f_small, x_small)
    grad_small_zyg = Zygote.gradient(f_small, x_small)[1]
    @info "compare gradients edge_cases_chainrules" mean(abs.(grad_small_fd .- grad_small_zyg))
    @test isapprox(grad_small_fd, grad_small_zyg, atol=1.0e-8)

    # Test with degree 0
    degree = 0
    f_degree0(x) = sum(create_basis(x, degree, Val(false)))
    grad_degree0_fd = ForwardDiff.gradient(f_degree0, x_small)

    # Degree 0 should give zero gradients (constant basis)
    @test all(iszero, grad_degree0_fd)

    # Test 1D input
    x_1d = rand(5)
    f_1d(x) = sum(create_basis(x, 2, Val(true)))
    grad_1d_fd = ForwardDiff.gradient(f_1d, x_1d)
    grad_1d_zyg = Zygote.gradient(f_1d, x_1d)[1]

    @test isapprox(grad_1d_fd, grad_1d_zyg, atol=1.0e-8)
end

# Currently borken at 1.12 bc of JET dependency.
# @testitem "comprehensive_ad_backend_test" begin
#     import Pkg
#     Pkg.add("Zygote")
#     Pkg.add("DifferentiationInterface")
#     Pkg.add("DifferentiationInterfaceTest")
#     Pkg.add("Mooncake")
#     Pkg.add("Enzyme")
#     Pkg.add("StableRNGs")

#     using DifferentiationInterface
#     using DifferentiationInterfaceTest
#     using StableRNGs
#     using ForwardDiff
#     using Zygote
#     using Mooncake
#     using Enzyme

#     # Setup test data
#     rng = StableRNG(1234)
#     N = 20
#     d_in = 2
#     x = rand(rng, N, d_in)
#     degree = 2

#     # Define test functions
#     f_basis_true(x_in) = sum(create_basis(x_in, degree, Val(true); center_data = true))
#     f_basis_false(x_in) = sum(create_basis(x_in, degree, Val(false)))
#     f_basis_default(x_in) = sum(create_basis(x_in, degree))

#     # Reference gradients using ForwardDiff
#     ∇f_basis_true = x -> ForwardDiff.gradient(f_basis_true, x)
#     ∇f_basis_false = x -> ForwardDiff.gradient(f_basis_false, x)
#     ∇f_basis_default = x -> ForwardDiff.gradient(f_basis_default, x)

#     # Define backends to test
#     backends = [AutoZygote(), AutoForwardDiff(), AutoMooncake(; config = nothing), AutoEnzyme()]

#     # Define scenarios
#     scenarios = [
#         Scenario{:gradient, :out}(f_basis_true, x; res1 = ∇f_basis_true(x)),
#         Scenario{:gradient, :out}(f_basis_false, x; res1 = ∇f_basis_false(x)),
#         Scenario{:gradient, :out}(f_basis_default, x; res1 = ∇f_basis_default(x)),
#     ]

#     # Run comprehensive tests
#     test_differentiation(
#         backends,
#         scenarios;
#         logging = false,
#         allocations = :none,
#         benchmark = :none,
#         correctness = true,
#         type_stability = :none,
#         detailed = true,
#         atol = 1.0e-3,
#         rtol = 1.0e-3,
#         count_calls = true,
#         scenario_intact = true
#     )
# end
