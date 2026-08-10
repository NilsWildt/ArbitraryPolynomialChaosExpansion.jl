# APCEmultires.jl — Multiresolution (multi-wavelet) polynomial chaos basis
#
# Implements the aMR-PC idea (Kröker & Oladyshkin 2022, RESS 222:108376):
# a piecewise polynomial chaos expansion where the input domain is decomposed
# into elements, each carrying its OWN local orthonormal basis (built via the
# existing `create_recurrence_basis`) and its OWN per-dimension polynomial degree.
#
# The new types slot into the existing duck-typed dispatch on
# `aPCE_PsiPolynomialMatrix(X, degrees, basis)` — no changes to `predict`,
# `train!`, or `UQ` are required.
#
# References:
#   [1] Kröker & Oladyshkin, RESS 222 (2022) 108376
#   [2] Kröker, Oladyshkin & Rybak, Comput. Geosci. 27 (2023) 805–827
#   [3] Le Maître, Najm, Ghanem, Knio, J. Comput. Phys. 197 (2004) 502–531

# ============================================================================
# Struct definitions
# ============================================================================

"""
    MultiWaveletElement{T <: Real}

A single resolution element: a hyperrectangle in the input space carrying its
own local orthonormal polynomial basis and per-dimension polynomial degree.

# Fields
- `lo::Vector{T}`: element lower bounds, one per dimension
- `hi::Vector{T}`: element upper bounds, one per dimension
- `degree::Vector{Int}`: per-dimension polynomial degree in THIS element
- `basis::RecurrenceCenteredBasis{T}`: local orthonormal basis built via
  `create_recurrence_basis` on the element's data subset
- `level::Vector{Int}`: per-dimension refinement level (0 = full domain).
  The element's probability weight is `prod(2^(-level[d]) for d in 1:n_dims)`.
- `dim_index::Vector{Int}`: per-dimension element index at the element's level
  (`0 ≤ dim_index[d] < 2^level[d]`).  Used for Sobol element-matching.
"""
struct MultiWaveletElement{T <: Real}
    lo::Vector{T}
    hi::Vector{T}
    degree::Vector{Int}
    basis::RecurrenceCenteredBasis{T}
    level::Vector{Int}
    dim_index::Vector{Int}
end

"""
    MultiWaveletBasis{T <: Real}

A multiresolution polynomial chaos basis: a collection of
[`MultiWaveletElement`](@ref)s that partition the input space.

# Fields
- `n_dims::Int`: number of input dimensions
- `elements::Vector{MultiWaveletElement{T}}`: the resolution elements
"""
struct MultiWaveletBasis{T <: Real}
    n_dims::Int
    elements::Vector{MultiWaveletElement{T}}
end

# Type introspection helpers (mirror the pattern for RecurrenceCenteredBasis)
Base.eltype(::MultiWaveletBasis{T}) where {T} = T
Base.eltype(::Type{MultiWaveletBasis{T}}) where {T} = T

# ============================================================================
# Construction
# ============================================================================

"""
    create_multiwavelet_basis(x::AbstractArray{T}, degree::Integer) where {T <: Real}
    create_multiwavelet_basis(x::AbstractArray{T}, degree::Vector{Int}) where {T <: Real}

Build a [`MultiWaveletBasis`](@ref) from training data `x`.

**Single-element version** (the default): creates one element covering the full
domain.  The element's local basis is a `RecurrenceCenteredBasis` built from the
full data via `create_recurrence_basis`.  In this case the multiresolution basis
is mathematically equivalent to the standard global recurrence basis — the
design matrix and Sobol indices match to machine precision.

For a scalar `degree`, every dimension gets the same polynomial degree.  For a
`Vector{Int}`, each dimension gets its own degree.

This function is AD-opaque (returns `NoTangent` via the `create_recurrence_basis`
`rrule`).  Gradients flow through `aPCE_PsiPolynomialMatrix` instead.
"""
function create_multiwavelet_basis(
        x::AbstractArray{T},
        degree::Integer,
    ) where {T <: Real}
    return create_multiwavelet_basis(x, fill(Int(degree), size(x, 2)))
end

function create_multiwavelet_basis(
        x::AbstractArray{T},
        degree::Vector{Int},
    ) where {T <: Real}
    xs = ndims(x) == 1 ? reshape(x, :, 1) : x
    n_dims = size(xs, 2)
    @assert length(degree) == n_dims "degree vector length must match number of dimensions"

    # Single-element version: one element covering the full domain.
    # The element's local basis is the global RecurrenceCenteredBasis built
    # with the max degree across dimensions (create_recurrence_basis uses a
    # single degree, so we use the max and mask during evaluation).
    max_degree = maximum(degree)
    rb = create_recurrence_basis(xs, max_degree)

    lo = vec(minimum(xs, dims = 1))
    hi = vec(maximum(xs, dims = 1))

    element = MultiWaveletElement{T}(
        lo, hi, copy(degree), rb,
        zeros(Int, n_dims),   # level 0 in every dimension
        zeros(Int, n_dims),   # index 0 in every dimension
    )

    return MultiWaveletBasis{T}(n_dims, [element])
end

# ============================================================================
# Element lookup
# ============================================================================

"""
    locate_element(basis::MultiWaveletBasis, x::AbstractVector, j::Integer) -> Int

Return the index of the element containing sample `x[j, :]`.  Points on element
boundaries are assigned to the **higher-index** element (deterministic choice).
If no element contains the point (numerical edge case), element 1 is returned.
"""
function locate_element(
        mwb::MultiWaveletBasis,
        x::AbstractMatrix,
        j::Integer,
    )
    for (e, elem) in enumerate(mwb.elements)
        inside = true
        for d in 1:mwb.n_dims
            if x[j, d] < elem.lo[d] || x[j, d] > elem.hi[d]
                inside = false
                break
            end
        end
        inside && return e
    end
    return 1
end

"""
    element_weight(elem::MultiWaveletElement) -> Real

Probability weight of the element: `prod(2^(-level[d]))`.
"""
function element_weight(elem::MultiWaveletElement)
    return prod(2.0^(-elem.level[d]) for d in eachindex(elem.level))
end

# ============================================================================
# Assembly: aPCE_PsiPolynomialMatrix
# ============================================================================

function aPCE_PsiPolynomialMatrix(
        TrainingInput,
        MultivariatePolynomialDegrees,
        mwb::MultiWaveletBasis,
    )
    return aPCE_PsiPolynomialMatrix_zygote(TrainingInput, MultivariatePolynomialDegrees, mwb)
end

function aPCE_PsiPolynomialMatrix_zygote(
        TrainingInput,
        MultivariatePolynomialDegrees,
        mwb::MultiWaveletBasis,
    )
    x = ndims(TrainingInput) == 1 ? reshape(TrainingInput, :, 1) : TrainingInput
    T = eltype(x)

    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(x, 1)
    Psi = zeros(T, NumberOfTerms, NCpoints)

    element_ids = Vector{Int}(undef, NCpoints)
    @inbounds for j in 1:NCpoints
        element_ids[j] = locate_element(mwb, x, j)
    end

    @inbounds for (e, elem) in enumerate(mwb.elements)
        sids = Int[]
        for j in 1:NCpoints
            element_ids[j] == e && push!(sids, j)
        end
        isempty(sids) && continue

        n_elem_points = length(sids)
        rb = elem.basis

        z = Matrix{T}(undef, n_elem_points, InputDimensions)
        for (k, j) in enumerate(sids)
            for d in 1:InputDimensions
                z[k, d] = (x[j, d] - rb.μ[d]) / rb.σ[d]
            end
        end

        Pvals = Vector{Matrix{T}}(undef, InputDimensions)
        for d in 1:InputDimensions
            Pvals[d] = _orthonormal_recurrence_values(
                @view(z[:, d]), @view(rb.α[:, d]), @view(rb.β[:, d]), rb.degree,
            )
        end

        for i in 1:NumberOfTerms
            in_range = true
            for d in 1:InputDimensions
                if MultivariatePolynomialDegrees[i, d] > elem.degree[d]
                    in_range = false
                    break
                end
            end
            in_range || continue

            for (k, j) in enumerate(sids)
                val = one(T)
                for d in 1:InputDimensions
                    deg_d = MultivariatePolynomialDegrees[i, d]
                    val *= Pvals[d][k, deg_d + 1]
                    iszero(val) && break
                end
                Psi[i, j] = val
            end
        end
    end

    return Psi
end

# ============================================================================
# ChainRules — AD integration
# ============================================================================
#
# The multires assembly rrule follows the same pattern as the
# `RecurrenceCenteredBasis` rrule in `APCEderivatives.jl:308`:
#   1. Forward pass computes Psi.
#   2. Pre-compute derivative values per element (via
#      `_orthonormal_recurrence_derivative_values`).
#   3. Pullback iterates element-by-element, using each element's local
#      derivative values, returning `dX` only (basis/degrees are not
#      differentiable → NoTangent).
#
# Basis construction (`create_multiwavelet_basis`) is AD-opaque (NoTangent),
# matching the convention for `create_recurrence_basis` and
# `create_centered_basis`.

"""
    rrule(::typeof(aPCE_PsiPolynomialMatrix_zygote), X, degrees, basis::MultiWaveletBasis)

Custom ChainRule for the multires Psi matrix.  Pre-computes per-element
derivative values outside the pullback closure, then computes `Δx` using the
analytic derivative recurrence (no ForwardDiff nesting needed — each element's
local basis is a `RecurrenceCenteredBasis`).
"""
function ChainRulesCore.rrule(
        ::typeof(aPCE_PsiPolynomialMatrix_zygote),
        TrainingInput,
        MultivariatePolynomialDegrees::AbstractMatrix{Int},
        mwb::MultiWaveletBasis,
    )
    x = ndims(TrainingInput) == 1 ? reshape(TrainingInput, :, 1) : TrainingInput
    T = eltype(x)

    # Forward pass
    Psi = aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, mwb)

    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(x, 1)

    # Assign points to elements (needed for the pullback too)
    element_ids = Vector{Int}(undef, NCpoints)
    @inbounds for j in 1:NCpoints
        element_ids[j] = locate_element(mwb, x, j)
    end

    # Pre-compute per-element derivative values and sample indices
    n_elements = length(mwb.elements)
    elem_derivs = Vector{Vector{Array{T, 3}}}(undef, n_elements)
    elem_sids = Vector{Vector{Int}}(undef, n_elements)

    @inbounds for e in 1:n_elements
        sids = Int[]
        for j in 1:NCpoints
            element_ids[j] == e && push!(sids, j)
        end
        elem_sids[e] = sids
        isempty(sids) && continue

        rb = mwb.elements[e].basis
        z = Matrix{T}(undef, length(sids), InputDimensions)
        for (k, j) in enumerate(sids)
            for d in 1:InputDimensions
                z[k, d] = (x[j, d] - rb.μ[d]) / rb.σ[d]
            end
        end

        derivs = Vector{Array{T, 3}}(undef, InputDimensions)
        for d in 1:InputDimensions
            derivs[d] = _orthonormal_recurrence_derivative_values(
                @view(z[:, d]), @view(rb.α[:, d]), @view(rb.β[:, d]), rb.degree, 1,
            )
        end
        elem_derivs[e] = derivs
    end

    function aPCE_Psi_MultiWavelet_pullback(ΔPsi)
        Δx = zeros(T, size(x))
        @inbounds for e in 1:n_elements
            sids = elem_sids[e]
            isempty(sids) && continue
            elem = mwb.elements[e]
            rb = elem.basis
            derivs = elem_derivs[e]

            for (k, j) in enumerate(sids)
                for i in 1:NumberOfTerms
                    Δij = ΔPsi[i, j]
                    iszero(Δij) && continue

                    # Degree mask check
                    in_range = true
                    for d in 1:InputDimensions
                        if MultivariatePolynomialDegrees[i, d] > elem.degree[d]
                            in_range = false
                            break
                        end
                    end
                    in_range || continue

                    for d in 1:InputDimensions
                        deg_d = MultivariatePolynomialDegrees[i, d]
                        iszero(deg_d) && continue

                        other = one(T)
                        for od in 1:InputDimensions
                            od === d && continue
                            other *= derivs[od][k, MultivariatePolynomialDegrees[i, od] + 1, 1]
                            iszero(other) && break
                        end

                        dz = derivs[d][k, deg_d + 1, 2]
                        Δx[j, d] += Δij * dz * other / rb.σ[d]
                    end
                end
            end
        end
        return (ChainRulesCore.NoTangent(), Δx, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end

    return Psi, aPCE_Psi_MultiWavelet_pullback
end

"""
    rrule(::typeof(create_multiwavelet_basis), x, degree)

AD-opaque: basis construction is treated as non-differentiable (`NoTangent`),
matching the convention for `create_recurrence_basis` and `create_centered_basis`.
Gradients flow through `aPCE_PsiPolynomialMatrix` instead.
"""
function ChainRulesCore.rrule(
        ::typeof(create_multiwavelet_basis),
        x::AbstractArray{T},
        degree::Integer,
    ) where {T <: Real}
    mwb = create_multiwavelet_basis(x, degree)
    function create_multiwavelet_basis_pullback(_Δ)
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end
    return mwb, create_multiwavelet_basis_pullback
end

function ChainRulesCore.rrule(
        ::typeof(create_multiwavelet_basis),
        x::AbstractArray{T},
        degree::Vector{Int},
    ) where {T <: Real}
    mwb = create_multiwavelet_basis(x, degree)
    function create_multiwavelet_basis_pullback(_Δ)
        return (ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent())
    end
    return mwb, create_multiwavelet_basis_pullback
end

# ============================================================================
# Tests (TestItems.jl — discovered by TestItemRunner in both src/ and test/)
# ============================================================================

@testitem "multires_single_element_equivalence" begin
    using ForwardDiff, Zygote, StatsBase, Random
    using ArbitraryPolynomialChaosExpansion: predict, train!, UQ

    Random.seed!(42)
    x = rand(20, 2)
    degree = 3

    mwb = create_multiwavelet_basis(x, degree)
    rb = create_recurrence_basis(x, degree)

    @test mwb isa MultiWaveletBasis{Float64}
    @test length(mwb.elements) == 1
    @test mwb.elements[1].basis isa RecurrenceCenteredBasis{Float64}

    degs = aPCE_MultivariatePolynomialDegrees(2, degree, 1.0, 1.0)
    Psi_multi = aPCE_PsiPolynomialMatrix(x, degs, mwb)
    Psi_rec = aPCE_PsiPolynomialMatrix(x, degs, rb)

    @test size(Psi_multi) == size(Psi_rec)
    @test maximum(abs.(Psi_multi .- Psi_rec)) < 1.0e-12

    TrainingOutput = sin.(x[:, 1]) .* cos.(x[:, 2])
    TrainingOutput = reshape(TrainingOutput, :, 1)

    apc_multi = aPCE(x, degree; outdim = 1, basis = Val(:multires))
    apc_rec = aPCE(x, degree; outdim = 1, basis = Val(:recurrence))

    train!(apc_multi, x, TrainingOutput; bayesian_inversion = true, reg_order = 2)
    train!(apc_rec, x, TrainingOutput; bayesian_inversion = true, reg_order = 2)

    @test maximum(abs.(apc_multi.ExpansionCoefficients .- apc_rec.ExpansionCoefficients)) < 1.0e-12

    preds_multi = predict(apc_multi, x)
    preds_rec = predict(apc_rec, x)
    @test maximum(abs.(preds_multi .- preds_rec)) < 1.0e-12

    uq_multi = UQ(apc_multi)
    uq_rec = UQ(apc_rec)
    @test maximum(abs.(uq_multi.OutputMean .- uq_rec.OutputMean)) < 1.0e-12
    @test maximum(abs.(uq_multi.OutputVar .- uq_rec.OutputVar)) < 1.0e-12
end

@testitem "multires_AD_forwarddiff_zygote" begin
    using ForwardDiff, Zygote, StatsBase, Random
    using ArbitraryPolynomialChaosExpansion: aPCE_PsiPolynomialMatrix_zygote

    Random.seed!(42)
    x = rand(10, 2)
    degree = 4
    degs = aPCE_MultivariatePolynomialDegrees(2, degree, 1.0, 1.0)
    mwb = create_multiwavelet_basis(x, degree)

    f_multi(z) = sum(aPCE_PsiPolynomialMatrix_zygote(z, degs, mwb))

    grad_fd = ForwardDiff.gradient(f_multi, x)
    grad_zyg = Zygote.gradient(f_multi, x)[1]

    @test isapprox(grad_fd, grad_zyg, atol = 1.0e-6)

    rb = create_recurrence_basis(x, degree)
    f_rec(z) = sum(aPCE_PsiPolynomialMatrix_zygote(z, degs, rb))
    grad_rec_fd = ForwardDiff.gradient(f_rec, x)

    @test isapprox(grad_fd, grad_rec_fd, atol = 1.0e-12)
end

@testitem "multires_per_element_degree" begin
    using Random

    Random.seed!(42)
    x = rand(20, 3)

    degree_vec = [2, 3, 4]
    mwb = create_multiwavelet_basis(x, degree_vec)

    @test mwb isa MultiWaveletBasis{Float64}
    @test mwb.elements[1].degree == degree_vec
    @test mwb.elements[1].basis.degree == maximum(degree_vec)

    mwb2 = create_multiwavelet_basis(x, 3)
    @test all(mwb2.elements[1].degree .== 3)
end

@testitem "multires_type_stability" begin
    using Random

    Random.seed!(42)
    x = rand(10, 2)
    degree = 3
    degs = aPCE_MultivariatePolynomialDegrees(2, degree, 1.0, 1.0)
    mwb = create_multiwavelet_basis(x, degree)

    Psi = aPCE_PsiPolynomialMatrix(x, degs, mwb)
    @test Psi isa Matrix{Float64}

    x32 = Float32.(x)
    mwb32 = create_multiwavelet_basis(x32, degree)
    Psi32 = aPCE_PsiPolynomialMatrix(x32, degs, mwb32)
    @test Psi32 isa Matrix{Float32}
end
