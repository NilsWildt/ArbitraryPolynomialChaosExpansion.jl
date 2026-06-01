"""
    APCEGradientOverrides

Custom AD rules for APCE functions that match MATLAB DaPC behavior.

The original MATLAB implementation treats the polynomial basis as CONSTANT during 
backpropagation - gradients only flow through the polynomial evaluation and weights,
NOT through the basis construction (orthonormalization).

This module provides wrapper functions with zero-gradient rules that match this behavior.

# Background

The APCE package (ArbitraryPolynomialChaosExpansion.jl) has rrules that DO differentiate
through `create_basis` using ForwardDiff. While mathematically correct, this causes 
numerical instability in deep networks because:

1. Orthonormalization involves Gram-Schmidt-like operations
2. Gradients through these operations can amplify small errors
3. In deep networks (3+ layers), this causes gradient explosion

The MATLAB code explicitly avoids this by treating basis as constant in backward():
```matlab
function [dLdInput, dLdWeights] = backward(layer, Input, ~, dLdOutput, Psi)
    % Psi is treated as CONSTANT here
    dLdWeights = dLdOutput * Psi';  % Only differentiate w.r.t. weights
    % dLdInput only considers polynomial evaluation derivatives
end
```

# Usage

```julia
using .APCEGradientOverrides: create_basis_constant

# In forward pass:
basis = create_basis_constant(x, degree; center_data=true)
# Gradients will NOT flow through this call
```
"""
module APCEGradientOverrides

using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @not_implemented
import ..create_basis, ..aPCE_PsiPolynomialMatrix_zygote, ..evalpoly_two

# Try to import Mooncake if available
const HAS_MOONCAKE = try
    using Mooncake: @from_rrule, DefaultCtx
    true
catch
    false
end

export create_basis_with_mode, ConstBasis, DiffBasis, psi_matrix_with_mode, psi_matrix_with_mode

# ============================================================================
# Trait System
# ============================================================================

abstract type AbstractBasisGradientMode end

"""
    ConstBasis <: AbstractBasisGradientMode

Treat basis as constant during backpropagation.
Gradients are BLOCKED through `create_basis`.
This is the recommended setting for deep APCE networks to prevent explosion.
"""
struct ConstBasis <: AbstractBasisGradientMode end

"""
    DiffBasis <: AbstractBasisGradientMode

Allow gradients to flow through basis construction.
This uses the standard AD rules from ArbitraryPolynomialChaosExpansion.jl.
"""
struct DiffBasis <: AbstractBasisGradientMode end

# ============================================================================
# Main Dispatch Function
# ============================================================================

"""
    create_basis_with_mode(mode, x, degree; center_data=true)

Create polynomial basis with specified gradient behavior.

# Arguments
- `mode`: `ConstBasis()` or `DiffBasis()`
- `x`: Input data array
- `degree`: Polynomial degree
- `center_data`: Whether to center data (default: true)
"""
function create_basis_with_mode(mode::AbstractBasisGradientMode, x::AbstractArray, degree::Integer; center_data::Bool = true)
    return create_basis(x, degree; center_data = center_data)
end

# Val dispatch wrapper
function create_basis_with_mode(mode::AbstractBasisGradientMode, x::AbstractArray, degree::Integer, ::Val{is_ortho}; center_data::Bool = true) where {is_ortho}
    return create_basis(x, degree, Val(is_ortho); center_data = center_data)
end

# ============================================================================
# Psi Matrix Evaluation with Gradient Control
# ============================================================================

"""
    psi_matrix_with_mode(mode, x, degrees, basis)

Compute Psi matrix with gradient behavior determined by mode.
For `ConstBasis`, uses standard APCE function (optimized, no gradients w.r.t basis).
For `DiffBasis`, uses differentiable implementation (slower, but full gradients).
"""
function psi_matrix_with_mode(::ConstBasis, x, degrees, basis)
    return aPCE_PsiPolynomialMatrix_zygote(x, degrees, basis)
end

function psi_matrix_with_mode(::DiffBasis, x, degrees, basis)
    # Differentiable implementation using map/stack for Zygote compatibility
    NumberOfTerms, InputDimensions = size(degrees)
    NCpoints = size(x, 1)

    rows = map(1:NumberOfTerms) do i
        map(1:NCpoints) do j
            term = one(eltype(x))
            for d in 1:InputDimensions
                degree = degrees[i, d]
                coeffs = @view basis[degree + 1, 1:(degree + 1), d]
                term *= evalpoly_two(x[j, d], coeffs)
            end
            return term
        end
    end

    return reduce(vcat, [transpose(r) for r in rows])
end

# ============================================================================
# ChainRulesCore rrules
# ============================================================================

# --- Rule for ConstBasis (Zero Gradients) ---

function ChainRulesCore.rrule(::typeof(create_basis_with_mode), ::ConstBasis, x::AbstractArray{T}, degree::Integer; center_data::Bool = true) where {T}
    basis = create_basis(x, degree; center_data = center_data)

    function create_basis_const_pullback(Δbasis)
        # Zero gradient for x, NoTangent for mode/degree
        return (NoTangent(), NoTangent(), ZeroTangent(), NoTangent())
    end

    return basis, create_basis_const_pullback
end

function ChainRulesCore.rrule(::typeof(create_basis_with_mode), ::ConstBasis, x::AbstractArray{T}, degree::Integer, ::Val{is_ortho}; center_data::Bool = true) where {T, is_ortho}
    basis = create_basis(x, degree, Val(is_ortho); center_data = center_data)

    function create_basis_const_pullback(Δbasis)
        return (NoTangent(), NoTangent(), ZeroTangent(), NoTangent(), NoTangent())
    end

    return basis, create_basis_const_pullback
end

# --- Rule for DiffBasis (Pass-through to standard AD) ---

# We don't need to define an rrule for DiffBasis!
# By NOT defining one, Zygote/ChainRules will differentiate through the body
# of `create_basis_with_mode`, which calls `create_basis`.
# `create_basis` has its own rrules in the APCE package, so those will be used.
# This gives us the "standard" behavior for free.

# ============================================================================
# Mooncake registration (if available)
# ============================================================================

# Register rules with Mooncake via @from_rrule macro
# This makes our ChainRulesCore rules available to Mooncake's AD system
if HAS_MOONCAKE
    @eval begin
        using Mooncake: @from_rrule, DefaultCtx

        # Register ConstBasis rule
        @from_rrule DefaultCtx Tuple{
            typeof(create_basis_with_mode),
            ConstBasis, AbstractArray, Integer,
        }

        @from_rrule DefaultCtx Tuple{
            typeof(create_basis_with_mode),
            ConstBasis, AbstractArray, Integer, Val,
        }

        # NOTE: For DiffBasis, Mooncake should just trace through since we have no custom rule
    end
end

# ============================================================================
# Tests
# ============================================================================

using TestItems: @testitem

@testitem "create_basis_with_mode - ConstBasis zero gradients" begin
    using ArbitraryPolynomialChaosExpansion.APCEGradientOverrides: create_basis_with_mode, ConstBasis
    using Zygote

    x = randn(Float32, 50, 3)
    degree = 3

    function loss_const(x_in)
        basis = create_basis_with_mode(ConstBasis(), x_in, degree)
        return sum(abs2, basis)
    end

    grad = Zygote.gradient(loss_const, x)[1]

    # Should be zero/nothing
    @test grad === nothing || all(iszero, grad)
end

@testitem "create_basis_with_mode - DiffBasis has gradients" begin
    using ArbitraryPolynomialChaosExpansion.APCEGradientOverrides: create_basis_with_mode, DiffBasis
    using Zygote

    x = randn(Float32, 50, 3)
    degree = 3

    function loss_diff(x_in)
        basis = create_basis_with_mode(DiffBasis(), x_in, degree)
        return sum(abs2, basis)
    end

    grad = Zygote.gradient(loss_diff, x)[1]

    # Should HAVE gradients
    @test grad !== nothing && !all(iszero, grad)
end

end # module
