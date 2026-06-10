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
                    coeffs = @view OrthonormalBasis[degree + 1, 1:(degree + 1), d]
                    @simd for k in 1:degree
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
function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray{T}, degree::Integer, ::Val{is_ortho}; center_data::Bool = true) where {T, is_ortho}
    basis = create_basis(x, degree, Val(is_ortho); center_data = center_data)

    function create_basis_pullback(Δbasis)
        function basis_wrapper(x_vec)
            x_reshaped = reshape(x_vec, size(x))
            basis_val = create_basis(x_reshaped, degree, Val(is_ortho); center_data = center_data)
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
function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray{T}, degree::Integer; center_data::Bool = true) where {T}
    return rrule(create_basis, x, degree, Val(true); center_data = center_data)
end

"""
    rrule(::typeof(create_basis), x, degree, is_orthonormal::Bool; center_data)

ChainRule for basis creation with Bool dispatch.
Converts Bool to Val for type-stable dispatch.
"""
function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray, degree::Integer, is_orthonormal::Bool; center_data::Bool = true)
    return rrule(create_basis, x, degree, Val(is_orthonormal); center_data = center_data)
end

"""
    rrule(::typeof(create_centered_basis), x, degree)

ChainRule for `create_centered_basis`. In the intended use pattern (e.g. the
IPCollocation layer), `x` is training data and not a trainable parameter;
the per-call basis is recomputed from the current batch but its gradient
w.r.t. `x` is not used by downstream optimization. We therefore treat `x` as
having no gradient contribution through basis construction — this makes the
call AD-opaque under Mooncake without forcing a ForwardDiff pass through the
Stieltjes / closed-form moment code (which is the instability source the
wrapper is designed to avoid).

Rationale recap: the closed-form coefficients use raw moments `m[k]` that
blow up with `σ^k` on un-scaled data; the Stieltjes back-transform via
binomial expansion is gradient-unfriendly. `CenteredBasis` keeps polynomial
coefficients in z-space and defers standardization to the Psi-matrix stage,
where the gradient path is plain broadcasting and therefore AD-stable.
"""
function ChainRulesCore.rrule(::typeof(create_centered_basis), x::AbstractArray{T}, degree::Integer) where {T}
    cb = create_centered_basis(x, degree)
    function create_centered_basis_pullback(_Δ)
        return (NoTangent(), NoTangent(), NoTangent())
    end
    return cb, create_centered_basis_pullback
end

"""
    rrule(::typeof(aPCE_OrthonormalBasis), Data, Degree, ::Val{center_data})

ChainRule for aPCE_OrthonormalBasis.
Uses ForwardDiff for gradient computation since this involves complex numerical operations.
"""
function ChainRulesCore.rrule(::typeof(aPCE_OrthonormalBasis), Data::AbstractArray{T}, Degree::Integer, center_data::Val{C}) where {T, C}
    basis = aPCE_OrthonormalBasis(Data, Degree, center_data)

    function aPCE_OrthonormalBasis_pullback(Δbasis)
        function basis_wrapper(data_vec)
            data_reshaped = reshape(data_vec, size(Data))
            basis_val = aPCE_OrthonormalBasis(data_reshaped, Degree, center_data)
            return sum(basis_val .* unthunk(Δbasis))
        end
        grad = ForwardDiff.gradient(basis_wrapper, vec(Data))
        grad_reshaped = reshape(grad, size(Data))
        return (NoTangent(), grad_reshaped, NoTangent(), NoTangent())
    end

    return basis, aPCE_OrthonormalBasis_pullback
end


# Enzyme.@import_rrule(typeof(create_basis), AbstractArray, Integer, Val)


ReverseDiff.@grad_from_chainrules create_basis(
    x::ReverseDiff.TrackedArray, d::Integer, is_orthonormal::Val{true}
);
ReverseDiff.@grad_from_chainrules create_basis(
    x::ReverseDiff.TrackedArray, d::Integer, is_orthonormal::Val{false}
);

# Note: Mooncake ChainRules integration is now in ext/MooncakeExt.jl
# It's loaded as a weak extension only when Mooncake is available


function ChainRulesCore.rrule(::typeof(solve_linear_robust), A, b; kwargs...)
    x = solve_linear_robust(A, b; kwargs...)
    function solve_linear_robust_pullback(dy)
        dy_unthunked = ChainRulesCore.unthunk(dy)
        λ = pinv(A') * dy_unthunked
        dA = -λ * x'
        db = λ
        return (ChainRulesCore.NoTangent(), dA, db)
    end
    return x, solve_linear_robust_pullback
end


# ===== TESTS =====

@testitem "create_basis_chainrules_consistency" begin
    using ForwardDiff, Zygote, StatsBase

    x = rand(20, 2)
    degree = 2

    # Test that all dispatch methods give same gradients
    f1(x) = sum(create_basis(x, degree))
    f2(x) = sum(create_basis(x, degree, Val(true)))
    f3(x) = sum(create_basis(x, degree, true))
    f4(x) = sum(create_basis(x, degree; center_data = true))

    grad1 = ForwardDiff.gradient(f1, x)
    grad2 = ForwardDiff.gradient(f2, x)
    grad3 = ForwardDiff.gradient(f3, x)
    grad4 = ForwardDiff.gradient(f4, x)
    @info "grad1: $grad1"
    @info "grad2: $grad2"
    @info "grad3: $grad3"
    @info "grad4: $grad4"
    @test isapprox(grad1, grad2, atol = 1.0e-10)
    @test isapprox(grad1, grad3, atol = 1.0e-10)
    @test isapprox(grad1, grad4, atol = 1.0e-10)

    # Test Zygote consistency
    grad1_zyg = Zygote.gradient(f1, x)[1]
    @test isapprox(grad1, grad1_zyg, atol = 1.0e-8)
end

@testitem "create_basis_differentiation_val_dispatch" begin
    using ForwardDiff, Zygote, StatsBase

    x = rand(15, 2)
    degree = 2

    # Test Val{true} dispatch
    f_true(x) = sum(create_basis(x, degree, Val(true); center_data = true))
    f_true_no_center(x) = sum(create_basis(x, degree, Val(true); center_data = false))

    grad_true_fd = ForwardDiff.gradient(f_true, x)
    grad_true_no_center_fd = ForwardDiff.gradient(f_true_no_center, x)

    grad_true_zyg = Zygote.gradient(f_true, x)[1]
    grad_true_no_center_zyg = Zygote.gradient(f_true_no_center, x)[1]

    @test isapprox(grad_true_fd, grad_true_zyg, atol = 1.0e-8)
    @test isapprox(grad_true_no_center_fd, grad_true_no_center_zyg, atol = 1.0e-8)

    # Test Val{false} dispatch
    f_false(x) = sum(create_basis(x, degree, Val(false)))

    grad_false_fd = ForwardDiff.gradient(f_false, x)
    grad_false_zyg = Zygote.gradient(f_false, x)[1]

    @test isapprox(grad_false_fd, grad_false_zyg, atol = 1.0e-8)
end

@testitem "create_basis_differentiation_bool_dispatch" begin
    using ForwardDiff, Zygote, StatsBase

    x = rand(12, 3)
    degree = 2

    # Test Bool dispatch
    f_bool_true(x) = sum(create_basis(x, degree, true; center_data = false))
    f_bool_false(x) = sum(create_basis(x, degree, false; center_data = true))

    grad_bool_true_fd = ForwardDiff.gradient(f_bool_true, x)
    grad_bool_false_fd = ForwardDiff.gradient(f_bool_false, x)

    grad_bool_true_zyg = Zygote.gradient(f_bool_true, x)[1]
    grad_bool_false_zyg = Zygote.gradient(f_bool_false, x)[1]

    @test isapprox(grad_bool_true_fd, grad_bool_true_zyg, atol = 1.0e-8)
    @test isapprox(grad_bool_false_fd, grad_bool_false_zyg, atol = 1.0e-8)

    # Test that Bool and Val give same results
    f_val_true(x) = sum(create_basis(x, degree, Val(true); center_data = false))
    f_val_false(x) = sum(create_basis(x, degree, Val(false); center_data = true))

    grad_val_true_fd = ForwardDiff.gradient(f_val_true, x)
    grad_val_false_fd = ForwardDiff.gradient(f_val_false, x)

    @test isapprox(grad_bool_true_fd, grad_val_true_fd, atol = 1.0e-12)
    @test isapprox(grad_bool_false_fd, grad_val_false_fd, atol = 1.0e-12)
end

@testitem "psi_matrix_chainrules" begin
    using ForwardDiff, Zygote, StatsBase
    # Setup test data
    x = rand(10, 2)
    degree = 2
    MultivariatePolynomialDegrees = aPCE_MultivariatePolynomialDegrees(2, degree, 1.0, 1.0)
    OrthonormalBasis = create_basis(x, degree, Val(true))

    # Test aPCE_PsiPolynomialMatrix
    f_psi(x) = sum(aPCE_PsiPolynomialMatrix(x, MultivariatePolynomialDegrees, OrthonormalBasis))

    grad_psi_fd = ForwardDiff.gradient(f_psi, x)
    grad_psi_zyg = Zygote.gradient(f_psi, x)[1]
    @info "compare gradients aPCE_PsiPolynomialMatrix" StatsBase.mean(abs.(grad_psi_fd .- grad_psi_zyg))
    @test isapprox(grad_psi_fd, grad_psi_zyg, atol = 1.0e-6)

    # Test aPCE_PsiPolynomialMatrix_zygote
    f_psi_zyg(x) = sum(ArbitraryPolynomialChaosExpansion.aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, OrthonormalBasis))

    grad_psi_zyg_fd = ForwardDiff.gradient(f_psi_zyg, x)
    grad_psi_zyg_zyg = Zygote.gradient(f_psi_zyg, x)[1]
    @info "compare gradients aPCE_PsiPolynomialMatrix_zygote" StatsBase.mean(abs.(grad_psi_zyg_fd .- grad_psi_zyg_zyg))

    @test isapprox(grad_psi_zyg_fd, grad_psi_zyg_zyg, atol = 1.0e-6)

    # Test that both methods give similar results
    @test isapprox(grad_psi_fd, grad_psi_zyg_fd, atol = 1.0e-6)
end


@testitem "reverse_columns_chainrules" begin
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
    using ForwardDiff, Zygote, StatsBase

    # Test with minimal data
    x_small = rand(2, 1)
    degree = 1

    f_small(x) = sum(create_basis(x, degree, Val(true)))
    grad_small_fd = ForwardDiff.gradient(f_small, x_small)
    grad_small_zyg = Zygote.gradient(f_small, x_small)[1]
    @info "compare gradients edge_cases_chainrules" StatsBase.mean(abs.(grad_small_fd .- grad_small_zyg))
    @test isapprox(grad_small_fd, grad_small_zyg, atol = 1.0e-8)

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

    @test isapprox(grad_1d_fd, grad_1d_zyg, atol = 1.0e-8)
end


@testitem "Mooncake autodiff - create_basis orthonormal" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ComponentArrays: ComponentArray
    using ArbitraryPolynomialChaosExpansion

    # Test gradient through orthonormal basis creation
    function loss_with_basis(x_flat, degree)
        x = reshape(x_flat, :, 2)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(true))
        return sum(abs2, basis)
    end

    backend = AutoMooncake()
    x = rand(Float32, 20)  # 10 samples × 2 dimensions
    degree = 3

    # Test type stability of gradient preparation
    # @test_nowarn @inferred prepare_gradient(loss_with_basis, backend, x, Constant(degree))

    # Prepare gradient
    extras = prepare_gradient(loss_with_basis, backend, x, Constant(degree))

    # Test gradient computation
    grad = similar(x)
    @test_nowarn gradient!(loss_with_basis, grad, extras, backend, x, Constant(degree))

    # Check gradient properties
    @test eltype(grad) == Float32
    @test length(grad) == length(x)
    @test all(isfinite, grad)
    @test !all(iszero, grad)  # Should have non-zero gradients

    # Test with Float64
    x64 = rand(Float64, 20)
    extras64 = prepare_gradient(loss_with_basis, backend, x64, Constant(degree))
    grad64 = similar(x64)
    gradient!(loss_with_basis, grad64, extras64, backend, x64, Constant(degree))

    @test eltype(grad64) == Float64
    @test all(isfinite, grad64)
end

@testitem "Mooncake autodiff - closed form basis degrees 0-4" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()

    # Test each closed-form degree separately
    for degree in 0:4
        x = rand(Float32, 50, 2)

        function loss_degree(x_mat)
            basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, degree, Val(true))
            return sum(abs2, basis)
        end

        # Test type inference
        # @test_nowarn @inferred prepare_gradient(loss_degree, backend, x)

        # Compute gradient
        extras = prepare_gradient(loss_degree, backend, x)
        grad = similar(x)
        @test_nowarn gradient!(loss_degree, grad, extras, backend, x)

        # Verify gradient properties
        @test size(grad) == size(x)
        @test eltype(grad) == Float32
        @test all(isfinite, grad)

        # Test value and gradient together
        val, grad2 = value_and_gradient(loss_degree, extras, backend, x)
        @test val isa Float32
        @test isfinite(val)
        @test grad2 ≈ grad
    end
end

@testitem "Mooncake autodiff - numerical basis degree > 4" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()

    # Test higher degrees that use numerical solver
    for degree in [5, 6, 7]
        x = rand(Float32, 100, 2)  # Need more samples for higher degrees

        function loss_high_degree(x_mat)
            basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, degree, Val(true))
            return sum(abs2, basis)
        end

        # Test type inference
        # @test_nowarn @inferred prepare_gradient(loss_high_degree, backend, x)

        # Compute gradient
        extras = prepare_gradient(loss_high_degree, backend, x)
        grad = similar(x)
        @test_nowarn gradient!(loss_high_degree, grad, extras, backend, x)

        # Verify gradient properties
        @test size(grad) == size(x)
        @test eltype(grad) == Float32
        @test all(isfinite, grad)
        @test !all(iszero, grad)
    end
end

@testitem "Mooncake autodiff - full basis (non-orthonormal)" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()
    x = rand(Float32, 30, 3)
    degree = 3

    function loss_full_basis(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, degree, Val(false))
        return sum(abs2, basis)
    end

    # Test type inference
    # @test_nowarn @inferred prepare_gradient(loss_full_basis, backend, x)

    # Compute gradient
    extras = prepare_gradient(loss_full_basis, backend, x)
    grad = similar(x)
    @test_nowarn gradient!(loss_full_basis, grad, extras, backend, x)

    # Full basis is constant (all 1s and 0s), so gradients should be zero
    @test all(iszero, grad)

    # Verify value computation works
    val = loss_full_basis(x)
    @test val isa Float32
    @test isfinite(val)
end

@testitem "Mooncake autodiff - type stability across types" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion
    backend = AutoMooncake()
    degree = 3

    function make_loss(T)
        function loss(x_flat)
            x = reshape(x_flat, :, 2)
            basis = ArbitraryPolynomialChaosExpansion.create_basis(x, degree, Val(true))
            return sum(abs2, basis)
        end
        return loss
    end

    # Test Float32
    loss32 = make_loss(Float32)
    x32 = rand(Float32, 20)
    extras32 = prepare_gradient(loss32, backend, x32)
    grad32 = similar(x32)

    gradient!(loss32, grad32, extras32, backend, x32)
    @test eltype(grad32) == Float32

    # Test Float64
    loss64 = make_loss(Float64)
    x64 = rand(Float64, 20)
    extras64 = prepare_gradient(loss64, backend, x64)
    grad64 = similar(x64)

    # @test_nowarn @inferred gradient!(loss64, grad64, extras64, backend, x64)
    @test eltype(grad64) == Float64
end

@testitem "Zygote autodiff - solve_levenberg_marquardt" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using LinearAlgebra
    using ErrorTypes

    backend = AutoMooncake()

    # Test gradient through Levenberg-Marquardt solver
    function loss_with_lm(A_flat, b)
        n = 4
        A = reshape(A_flat, n, n)
        A_symm = A' * A + I  # Make positive definite
        x = unwrap(ArbitraryPolynomialChaosExpansion.solve_levenberg_marquardt(A_symm, b))
        return sum(abs2, x)
    end

    A_flat = rand(Float32, 16)
    b = rand(Float32, 4)

    # Test type inference
    # @test_nowarn @inferred prepare_gradient(loss_with_lm, backend, A_flat, Constant(b))

    # Compute gradient
    extras = prepare_gradient(loss_with_lm, backend, A_flat, Constant(b))
    grad = similar(A_flat)
    @test_nowarn gradient!(loss_with_lm, grad, extras, backend, A_flat, Constant(b))

    # Verify gradient properties
    @test eltype(grad) == Float32
    @test length(grad) == 16
    @test all(isfinite, grad)

    # Test value and gradient
    val, grad2 = value_and_gradient(loss_with_lm, extras, backend, A_flat, Constant(b))
    @test val isa Float32
    @test grad2 ≈ grad
end


@testitem "Mooncake autodiff - solve_levenberg_marquardt" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using LinearAlgebra
    using ErrorTypes
    using ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()

    # Test gradient through Levenberg-Marquardt solver
    function loss_with_lm(A_flat, b)
        n = 4
        A = reshape(A_flat, n, n)
        A_symm = A' * A + I  # Make positive definite
        x = unwrap(ArbitraryPolynomialChaosExpansion.solve_levenberg_marquardt(A_symm, b))
        return sum(abs2, x)
    end

    A_flat = rand(Float32, 16)
    b = rand(Float32, 4)

    # Test type inference
    # @test_nowarn @inferred prepare_gradient(loss_with_lm, backend, A_flat, Constant(b))

    # Compute gradient
    extras = prepare_gradient(loss_with_lm, backend, A_flat, Constant(b))
    grad = similar(A_flat)
    @test_nowarn gradient!(loss_with_lm, grad, extras, backend, A_flat, Constant(b))

    # Verify gradient properties
    @test eltype(grad) == Float32
    @test length(grad) == 16
    @test all(isfinite, grad)

    # Test value and gradient
    val, grad2 = value_and_gradient(loss_with_lm, extras, backend, A_flat, Constant(b))
    @test val isa Float32
    @test grad2 ≈ grad
end

@testitem "Mooncake autodiff - gradient numerical accuracy" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using FiniteDifferences
    using ArbitraryPolynomialChaosExpansion

    backend_mooncake = AutoMooncake()
    backend_fd = AutoFiniteDifferences(; fdm = FiniteDifferences.central_fdm(5, 1))

    x = rand(Float64, 30, 2)
    degree = 3

    function loss(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, degree, Val(true))
        return sum(abs2, basis)
    end

    # Compute Mooncake gradient
    extras_moon = prepare_gradient(loss, backend_mooncake, x)
    grad_moon = gradient(loss, extras_moon, backend_mooncake, x)

    # Compute finite differences gradient
    extras_fd = prepare_gradient(loss, backend_fd, x)
    grad_fd = gradient(loss, extras_fd, backend_fd, x)

    # Should match within reasonable tolerance
    @test grad_moon ≈ grad_fd rtol = 1.0e-4
end

@testitem "Mooncake autodiff - center_data parameter" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()
    x = rand(Float32, 50, 2)
    degree = 6  # Use numerical solver

    # Test with center_data=true
    function loss_centered(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(
            x_mat, degree, Val(true); center_data = true
        )
        return sum(abs2, basis)
    end

    extras_centered = prepare_gradient(loss_centered, backend, x)
    grad_centered = gradient(loss_centered, extras_centered, backend, x)

    @test all(isfinite, grad_centered)

    # Test with center_data=false
    function loss_not_centered(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(
            x_mat, degree, Val(true); center_data = false
        )
        return sum(abs2, basis)
    end

    extras_not_centered = prepare_gradient(loss_not_centered, backend, x)
    grad_not_centered = gradient(loss_not_centered, extras_not_centered, backend, x)

    @test all(isfinite, grad_not_centered)

    # Gradients should be different
    @test !isapprox(grad_centered, grad_not_centered, rtol = 1.0e-3)
end

@testitem "CenteredBasis - Ψ round-trip (x == z-scored + cb.basis)" begin
    using ArbitraryPolynomialChaosExpansion
    const APCE = ArbitraryPolynomialChaosExpansion

    x = rand(20, 3) .* 5.0 .+ 2.0
    cb = APCE.create_centered_basis(x, 3)
    degs = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 2 0 0; 0 2 0; 0 0 2]

    # Dispatched path: standardizes x internally, then delegates to AbstractArray method
    Ψ_cb = APCE.aPCE_PsiPolynomialMatrix_zygote(x, degs, cb)

    # Manual path: standardize outside, call AbstractArray method with cb.basis
    μrow = reshape(cb.μ, 1, :)
    σrow = reshape(cb.σ, 1, :)
    z = (x .- μrow) ./ σrow
    Ψ_direct = APCE.aPCE_PsiPolynomialMatrix_zygote(z, degs, cb.basis)

    @test maximum(abs, Ψ_cb .- Ψ_direct) < 1.0e-12
    @test size(cb.basis) == (4, 4, 3)
    @test length(cb.μ) == 3 && length(cb.σ) == 3
    @test all(isfinite, Ψ_cb)
end

@testitem "CenteredBasis - Mooncake gradient vs finite differences (fixed basis)" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using FiniteDifferences
    using ArbitraryPolynomialChaosExpansion
    const APCE = ArbitraryPolynomialChaosExpansion

    backend_mc = AutoMooncake()
    backend_fd = AutoFiniteDifferences(; fdm = FiniteDifferences.central_fdm(5, 1))

    # Fit basis from a reference sample; held constant in the loss closure
    x_ref = rand(50, 3) .* 3.0 .+ 1.0
    cb = APCE.create_centered_basis(x_ref, 3)
    degs = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 2 0 0; 0 2 0; 0 0 2; 1 1 0; 1 0 1]

    x = rand(8, 3) .* 2.0 .+ 0.5

    # Differentiate only through the Ψ-evaluation path; cb is constant.
    # This is the mathematically well-defined test — basis is treated as data.
    loss(xlocal) = sum(abs2, APCE.aPCE_PsiPolynomialMatrix_zygote(xlocal, degs, cb))

    extras_mc = prepare_gradient(loss, backend_mc, x)
    grad_mc = gradient(loss, extras_mc, backend_mc, x)

    extras_fd = prepare_gradient(loss, backend_fd, x)
    grad_fd = gradient(loss, extras_fd, backend_fd, x)

    @test grad_mc ≈ grad_fd rtol = 1.0e-5
    @test all(isfinite, grad_mc)
end

@testitem "CenteredBasis - create_centered_basis is AD-opaque (NoTangent rrule)" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion
    const APCE = ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()
    x = rand(15, 3) .* 4.0 .+ 1.0
    degs = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 2 0 0; 0 2 0]

    # lossA: basis rebuilt inside the differentiated function (opaque path used)
    function lossA(xlocal)
        cb = APCE.create_centered_basis(xlocal, 3)
        return sum(abs2, APCE.aPCE_PsiPolynomialMatrix_zygote(xlocal, degs, cb))
    end

    # lossB: basis held constant in closure (no opaque call on the AD tape)
    cb_fixed = APCE.create_centered_basis(x, 3)
    lossB(xlocal) = sum(abs2, APCE.aPCE_PsiPolynomialMatrix_zygote(xlocal, degs, cb_fixed))

    # At the reference point, both evaluations must match
    @test lossA(x) ≈ lossB(x)

    extras_A = prepare_gradient(lossA, backend, x)
    grad_A = gradient(lossA, extras_A, backend, x)

    extras_B = prepare_gradient(lossB, backend, x)
    grad_B = gradient(lossB, extras_B, backend, x)

    # NoTangent rrule on create_centered_basis means it contributes zero to the
    # gradient. Only the explicit Ψ-evaluation path matters, so grad_A == grad_B.
    @test grad_A ≈ grad_B atol = 1.0e-10
    @test all(isfinite, grad_A)
end

@testitem "Mooncake autodiff - edge cases" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()

    # Test degree 0 (should be trivial)
    x0 = rand(Float32, 10, 1)
    function loss_deg0(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, 0, Val(true))
        return sum(abs2, basis)
    end

    extras0 = prepare_gradient(loss_deg0, backend, x0)
    grad0 = gradient(loss_deg0, extras0, backend, x0)
    @test all(iszero, grad0)  # Constant basis → zero gradient

    # Test single dimension
    x1d = rand(Float32, 50, 1)
    function loss_1d(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, 3, Val(true))
        return sum(abs2, basis)
    end

    extras1d = prepare_gradient(loss_1d, backend, x1d)
    grad1d = gradient(loss_1d, extras1d, backend, x1d)
    @test size(grad1d) == size(x1d)
    @test all(isfinite, grad1d)

    # Test many dimensions
    x_multi = rand(Float32, 30, 5)
    function loss_multi(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, 2, Val(true))
        return sum(abs2, basis)
    end

    extras_multi = prepare_gradient(loss_multi, backend, x_multi)
    grad_multi = gradient(loss_multi, extras_multi, backend, x_multi)
    @test size(grad_multi) == size(x_multi)
    @test all(isfinite, grad_multi)
end

@testitem "Mooncake autodiff - check_mode compatibility" begin
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion

    backend = AutoMooncake()
    x = rand(Float32, 40, 2)
    degree = 4

    function loss(x_mat)
        basis = ArbitraryPolynomialChaosExpansion.create_basis(x_mat, degree, Val(true))
        return sum(abs2, basis)
    end

    # Verify mode support
    @test check_available(backend)

    # Test gradient mode
    @test gradient(loss, backend, x) isa typeof(x)

    # Test value_and_gradient mode
    val, grad = value_and_gradient(loss, backend, x)
    @test val isa Float32
    @test grad isa typeof(x)
    @test all(isfinite, grad)
end


# Disabled: this testitem calls Pkg.add at runtime. JET itself works on Julia
# 1.12 and is exercised in test/test_jet.jl.
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
