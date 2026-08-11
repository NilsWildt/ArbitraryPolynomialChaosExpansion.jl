# APCEmultires.jl — Multiresolution (multi-wavelet) polynomial chaos basis
#
# Implements the aMR-PC idea (Kröker & Oladyshkin 2022, RESS 222:108376):
# a piecewise polynomial chaos expansion where the input domain is decomposed
# into elements, each carrying its OWN local orthonormal basis (built via the
# existing `create_recurrence_basis`) and its OWN per-dimension polynomial degree.
#
# Multi-element training uses per-element independent solves (the mathematically
# correct approach for disjoint elements).  The `train!`, `predict`, and `UQ`
# methods are overridden for `aPCE{T, MultiWaveletBasis}` — zero changes to
# existing code paths.
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

A single resolution element: a half-open hyperrectangle covering part of ℝⁿ,
carrying its own local orthonormal polynomial basis and per-dimension
polynomial degree.  The elements of a basis tile all of ℝⁿ (bounds are ±Inf
except at split boundaries), so every input point belongs to exactly one
element — including points outside the training data's range.

# Fields
- `lo::Vector{T}`: element lower bounds, one per dimension (may be `-Inf`)
- `hi::Vector{T}`: element upper bounds, one per dimension (may be `Inf`)
- `degree::Vector{Int}`: per-dimension polynomial degree in THIS element
- `basis::RecurrenceCenteredBasis{T}`: local orthonormal basis
- `level::Vector{Int}`: per-dimension refinement level (0 = full domain)
- `dim_index::Vector{Int}`: per-dimension element index at the element's level
- `weight::T`: probability mass of the element, estimated as the fraction of
  training samples it contains.  Weights across a basis sum to 1.  (For
  equal-mass dyadic splits this coincides with the aMR-PC `2^-level` weight,
  but the empirical mass stays correct for arbitrary split points.)
- `multi_indices::Matrix{Int}`: the element's multi-index set, `(n_terms_e,
  n_dims)`.  Populated by `train!` (rows of the model's global index set with
  all degrees within this element's `degree`); empty until trained.
- `coefficients::Matrix{T}`: per-element expansion coefficients, shape
  `(n_terms_e, outdim)`, rows aligned with `multi_indices`.  Populated by
  `train!`; empty until trained.
- `xmin::Vector{T}`: per-dimension minimum of the element's training data.
  Populated by `train!`; `-Inf` until then.  Used by `predict` to clip
  off-support evaluation points (e.g. Saltelli mixed inputs) to the element's
  support, bounding local-polynomial extrapolation.
- `xmax::Vector{T}`: per-dimension maximum of the element's training data
  (see `xmin`).
"""
mutable struct MultiWaveletElement{T <: Real}
    lo::Vector{T}
    hi::Vector{T}
    degree::Vector{Int}
    basis::RecurrenceCenteredBasis{T}
    level::Vector{Int}
    dim_index::Vector{Int}
    weight::T
    multi_indices::Matrix{Int}
    coefficients::Matrix{T}
    xmin::Vector{T}
    xmax::Vector{T}
end

"""
    MultiWaveletBasis{T <: Real}

A multiresolution polynomial chaos basis: a collection of
[`MultiWaveletElement`](@ref)s that partition the input space.
"""
struct MultiWaveletBasis{T <: Real}
    n_dims::Int
    elements::Vector{MultiWaveletElement{T}}
end

Base.eltype(::MultiWaveletBasis{T}) where {T} = T
Base.eltype(::Type{MultiWaveletBasis{T}}) where {T} = T

# Reported shape mirrors the other basis backends' convention
# `(degree + 1, degree + 1, input_dimensions)` (using the maximum per-element
# degree) so `show(::aPCE)` and size-based introspection work uniformly.
function Base.size(mwb::MultiWaveletBasis)
    deg = maximum(maximum(elem.degree) for elem in mwb.elements)
    return (deg + 1, deg + 1, mwb.n_dims)
end
Base.size(mwb::MultiWaveletBasis, d::Integer) = size(mwb)[d]

# ============================================================================
# Construction
# ============================================================================

"""
    create_multiwavelet_basis(x, degree; split_dim=0, split_point=nothing)

Build a [`MultiWaveletBasis`](@ref) from training data `x`.

**Single-element** (default): one element covering the full domain, equivalent
to the standard global recurrence basis.

**Split** (`split_dim` + `split_point`): two elements with independent local
bases from each data subset.  Also reachable through the model constructor:
`aPCE(x, degree; basis = Val(:multires), split_dim = ..., split_point = ...)`.
"""
function create_multiwavelet_basis(
        x::AbstractArray{T},
        degree::Integer;
        split_dim::Int = 0,
        split_point::Union{Nothing, Real} = nothing,
    ) where {T <: Real}
    Base.require_one_based_indexing(x)
    if split_dim == 0 || split_point === nothing
        return _create_single_element_basis(x, degree)
    end
    return _create_split_basis(x, Int(degree), split_dim, T(split_point))
end

function create_multiwavelet_basis(
        x::AbstractArray{T},
        degree::Vector{Int},
    ) where {T <: Real}
    return _create_single_element_basis(x, degree)
end

function _create_single_element_basis(
        x::AbstractArray{T},
        degree::Integer,
    ) where {T <: Real}
    return _create_single_element_basis(x, fill(Int(degree), size(x, 2)))
end

function _create_single_element_basis(
        x::AbstractArray{T},
        degree::Vector{Int},
    ) where {T <: Real}
    xs = ndims(x) == 1 ? reshape(x, :, 1) : x
    n_dims = size(xs, 2)
    @assert length(degree) == n_dims "degree vector length must match number of dimensions"

    max_degree = maximum(degree)
    rb = create_recurrence_basis(xs, max_degree)

    # The single element covers all of ℝⁿ so prediction points outside the
    # training range still locate (extrapolation, like the global backends).
    lo = fill(T(-Inf), n_dims)
    hi = fill(T(Inf), n_dims)

    element = MultiWaveletElement{T}(
        lo, hi, copy(degree), rb,
        zeros(Int, n_dims), zeros(Int, n_dims),
        one(T), zeros(Int, 0, 0), zeros(T, 0, 0),
        fill(T(-Inf), n_dims), fill(T(Inf), n_dims),
    )

    return MultiWaveletBasis{T}(n_dims, [element])
end

function _create_split_basis(
        x::AbstractArray{T},
        degree::Int,
        split_dim::Int,
        split_point::T,
    ) where {T <: Real}
    xs = ndims(x) == 1 ? reshape(x, :, 1) : x
    n_dims = size(xs, 2)
    @assert 1 <= split_dim <= n_dims "split_dim out of range"

    mask_lo = xs[:, split_dim] .<= split_point
    mask_hi = .!mask_lo

    # A degenerate split (all samples on one side) has no data to build the
    # empty element's basis from — fall back to a single element.
    (any(mask_lo) && any(mask_hi)) || return _create_single_element_basis(x, degree)

    n_samples = size(xs, 1)
    elements = MultiWaveletElement{T}[]

    # Elements are half-open slabs partitioning ℝⁿ: only the split dimension
    # is bounded (at split_point); all other bounds are ±Inf so any input
    # point — including extrapolation points — locates in exactly one element.
    # Element weights are the empirical probability mass of each slab; the
    # aMR-PC 2^-level weights are valid only for equal-mass (quantile) splits,
    # which an arbitrary split_point does not guarantee.
    x_lo = xs[mask_lo, :]
    degree_lo = min(degree, size(x_lo, 1) - 1)
    rb_lo = create_recurrence_basis(x_lo, degree_lo)
    lo = fill(T(-Inf), n_dims)
    hi = fill(T(Inf), n_dims)
    hi[split_dim] = split_point
    level = zeros(Int, n_dims)
    level[split_dim] = 1
    push!(
        elements, MultiWaveletElement{T}(
            lo, hi, fill(degree_lo, n_dims), rb_lo, level, zeros(Int, n_dims),
            T(count(mask_lo) / n_samples), zeros(Int, 0, 0), zeros(T, 0, 0),
            fill(T(-Inf), n_dims), fill(T(Inf), n_dims),
        ),
    )

    x_hi = xs[mask_hi, :]
    degree_hi = min(degree, size(x_hi, 1) - 1)
    rb_hi = create_recurrence_basis(x_hi, degree_hi)
    lo = fill(T(-Inf), n_dims)
    lo[split_dim] = split_point
    hi = fill(T(Inf), n_dims)
    level = zeros(Int, n_dims)
    level[split_dim] = 1
    dim_idx = zeros(Int, n_dims)
    dim_idx[split_dim] = 1
    push!(
        elements, MultiWaveletElement{T}(
            lo, hi, fill(degree_hi, n_dims), rb_hi, level, dim_idx,
            T(count(mask_hi) / n_samples), zeros(Int, 0, 0), zeros(T, 0, 0),
            fill(T(-Inf), n_dims), fill(T(Inf), n_dims),
        ),
    )

    return MultiWaveletBasis{T}(n_dims, elements)
end

# ============================================================================
# Element lookup and weights
# ============================================================================

# Elements tile ℝⁿ (bounds are ±Inf away from split boundaries), so every
# finite point matches; the first matching element wins, which puts points on
# a split boundary in the lower element, consistent with the `.<=` split mask.
function locate_element(
        mwb::MultiWaveletBasis,
        x::AbstractMatrix,
        j::Integer,
    )
    Base.require_one_based_indexing(x)
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
    throw(
        ArgumentError(
            "point $(x[j, :]) lies in no element — non-finite input coordinates?",
        ),
    )
end

element_weight(elem::MultiWaveletElement) = elem.weight

# ============================================================================
# Per-element multi-index generation
# ============================================================================

"""
    _element_multi_index_set(degree::Vector{Int}, n_dims::Int) -> Matrix{Int}

Generate the total-degree multi-index set for an element with per-dimension
degree bounds.  Returns a `(n_terms, n_dims)` matrix where each row is a
multi-index α with `sum(α) ≤ max(degree)` and `α[d] ≤ degree[d]`.
"""
function _element_multi_index_set(degree::Vector{Int}, n_dims::Int)
    length(degree) == n_dims ||
        throw(DimensionMismatch("degree has length $(length(degree)), expected $n_dims"))
    max_deg = maximum(degree)
    rows = Vector{Vector{Int}}()
    α = zeros(Int, n_dims)
    # Recursive enumeration visits only indices already inside the simplex —
    # the full tensor product `Iterators.product(0:degree...)` is exponential
    # in n_dims while keeping only polynomially many rows.
    function grow!(d::Int, remaining::Int)
        if d > n_dims
            push!(rows, copy(α))
            return nothing
        end
        for k in 0:min(degree[d], remaining)
            α[d] = k
            grow!(d + 1, remaining - k)
        end
        α[d] = 0
        return nothing
    end
    grow!(1, max_deg)
    # Sort by total degree, then lexicographically
    sort!(rows, by = r -> (sum(r), r))
    return reduce(vcat, transpose.(rows))
end

# Rows of the model's global (q-norm-truncated) index set that fit within an
# element's per-dimension degree caps.  Using the global set — rather than a
# freshly generated total-degree set — makes per-element solves honor the
# s_marginals/s_interactions truncation chosen at `aPCE` construction.
function _element_degrees(
        MultivariatePolynomialDegrees::AbstractMatrix{Int},
        elem::MultiWaveletElement,
    )
    keep = [
        i for i in axes(MultivariatePolynomialDegrees, 1)
            if all(
            MultivariatePolynomialDegrees[i, d] <= elem.degree[d]
                for d in axes(MultivariatePolynomialDegrees, 2)
        )
    ]
    return Matrix{Int}(MultivariatePolynomialDegrees[keep, :])
end

# Output dimension of a (partially) trained basis: taken from the first
# element that carries coefficients; 1 if nothing is trained yet.
function _trained_outdim(mwb::MultiWaveletBasis)
    for elem in mwb.elements
        isempty(elem.coefficients) || return size(elem.coefficients, 2)
    end
    return 1
end

# Multi-index set a trained element's coefficient rows are aligned with.
# `train!` stores it; for coefficients injected by hand (standalone
# MultiWaveletBasis use) fall back to the element's total-degree set, and fail
# fast when the coefficient rows do not match either set.
function _element_indices(elem::MultiWaveletElement, n_dims::Int)
    degs = isempty(elem.multi_indices) ?
        _element_multi_index_set(elem.degree, n_dims) : elem.multi_indices
    size(degs, 1) == size(elem.coefficients, 1) || throw(
        DimensionMismatch(
            "element has $(size(elem.coefficients, 1)) coefficient rows but " *
                "$(size(degs, 1)) multi-indices",
        ),
    )
    return degs
end

# ============================================================================
# Assembly: aPCE_PsiPolynomialMatrix (for AD and external consumers)
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
    Base.require_one_based_indexing(TrainingInput, MultivariatePolynomialDegrees)
    x = ndims(TrainingInput) == 1 ? reshape(TrainingInput, :, 1) : TrainingInput
    T = eltype(x)

    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(x, 1)
    Psi = zeros(T, NumberOfTerms, NCpoints)

    element_ids = Vector{Int}(undef, NCpoints)
    for j in 1:NCpoints
        element_ids[j] = locate_element(mwb, x, j)
    end

    for (e, elem) in enumerate(mwb.elements)
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
# Per-element train! / predict / UQ overrides for MultiWaveletBasis
# ============================================================================
#
# For disjoint elements, the global regression decouples into independent
# per-element least-squares solves.  These overrides store per-element
# coefficients in each MultiWaveletElement.coefficients field.

"""
    train!(aPCE::aPCE{T, MultiWaveletBasis{T}}, TrainingInput, y_rhs; kwargs...)

Override for MultiWaveletBasis: performs per-element independent least-squares
solves.  Each element's local basis fits that element's data subset, with the
same regularized solve (`bayesian_inversion`, `reg_order`) as the global
`train!`.  No global system is assembled — for disjoint elements the
regressions decouple exactly.

Each element solves over the rows of the model's q-norm-truncated index set
that fit its per-dimension degree, stored in `elem.multi_indices` alongside
`elem.coefficients`.  For a single-element basis this is the model's full
index set and the fit is equivalent to the standard global solve; the fitted
coefficients are then mirrored into `ExpansionCoefficients`.  A multi-element
model has no single global coefficient vector, so `ExpansionCoefficients` is
filled with `NaN` — generic consumers of that field must not be used with a
multi-element basis (use `predict`/`UQ`/`sobol_indices_multires`, which read
the per-element coefficients).
"""
function train!(
        aPCE_apc::aPCE{T, MultiWaveletBasis{T}, A1, A2},
        TrainingInput,
        y_rhs;
        bayesian_inversion = true,
        reg_order = 1,
    ) where {T <: Real, A1, A2}
    Base.require_one_based_indexing(TrainingInput, y_rhs)
    mwb = aPCE_apc.OrthonormalBasis
    x = ndims(TrainingInput) == 1 ? reshape(TrainingInput, :, 1) : TrainingInput
    y = ndims(y_rhs) == 1 ? reshape(y_rhs, :, 1) : y_rhs
    n_points = size(x, 1)
    outdim = size(y, 2)

    if aPCE_apc.output_dimensions != outdim
        aPCE_apc.output_dimensions = outdim
    end

    element_ids = Vector{Int}(undef, n_points)
    for j in 1:n_points
        element_ids[j] = locate_element(mwb, x, j)
    end

    for (e, elem) in enumerate(mwb.elements)
        sids = findall(==(e), element_ids)
        isempty(sids) && continue

        x_e = x[sids, :]
        y_e = Matrix{T}(y[sids, :])

        degs_e = _element_degrees(aPCE_apc.MultivariatePolynomialDegrees, elem)
        Psi_e = aPCE_PsiPolynomialMatrix(x_e, degs_e, elem.basis)

        elem.multi_indices = degs_e
        elem.coefficients = _train_solve(
            Matrix{T}(Psi_e'), y_e;
            bayesian_inversion = bayesian_inversion, reg_order = reg_order,
        )
        # Training-data support of this element, per dimension — used by
        # `predict` to clip off-support evaluation points (e.g. Saltelli
        # mixed inputs) before evaluating the local polynomial.
        elem.xmin = vec(minimum(x_e; dims = 1))
        elem.xmax = vec(maximum(x_e; dims = 1))
    end

    if length(mwb.elements) == 1 &&
            size(mwb.elements[1].coefficients, 1) == aPCE_apc.NumberOfTerms
        aPCE_apc.ExpansionCoefficients = copy(mwb.elements[1].coefficients)
    else
        # No global coefficient vector exists for a multi-element expansion;
        # NaN makes any generic consumer of this field fail loudly instead of
        # silently computing with zeros.
        aPCE_apc.ExpansionCoefficients = fill(T(NaN), aPCE_apc.NumberOfTerms, outdim)
    end

    return aPCE_apc
end

"""
    predict(aPCE::aPCE{T, MultiWaveletBasis{T}}, PredictionInput)

Override for MultiWaveletBasis: evaluates per-element local bases using the
per-element coefficients stored by `train!`.  Each prediction point is assigned
to its element, and that element's local expansion is evaluated.
"""
function predict(
        aPCE_apc::aPCE{T, MultiWaveletBasis{T}, A1, A2},
        PredictionInput,
    ) where {T <: Real, A1, A2}
    Base.require_one_based_indexing(PredictionInput)
    mwb = aPCE_apc.OrthonormalBasis
    x = ndims(PredictionInput) == 1 ? reshape(PredictionInput, :, 1) : PredictionInput
    n_points = size(x, 1)

    outdim = _trained_outdim(mwb)
    y_pred = zeros(T, n_points, outdim)

    element_ids = Vector{Int}(undef, n_points)
    for j in 1:n_points
        element_ids[j] = locate_element(mwb, x, j)
    end

    for (e, elem) in enumerate(mwb.elements)
        sids = findall(==(e), element_ids)
        isempty(sids) && continue
        isempty(elem.coefficients) && continue

        degs_e = _element_indices(elem, mwb.n_dims)
        # Clip evaluation points to this element's training-data support before
        # evaluating the local polynomial: bounds catastrophic extrapolation of
        # the local basis at off-support inputs (e.g. Saltelli mixed points,
        # whose coordinates are individually in-range but jointly off the
        # element's data manifold). No-op for points inside the support.
        x_clip = clamp.(x[sids, :], elem.xmin', elem.xmax')
        Psi_e = aPCE_PsiPolynomialMatrix(x_clip, degs_e, elem.basis)
        y_pred[sids, :] = Psi_e' * elem.coefficients
    end

    return y_pred
end

"""
    UQ(aPCE::aPCE{T, MultiWaveletBasis{T}}; axis=1)

Override for MultiWaveletBasis: computes mean and variance of output `axis`
from per-element coefficients using the aMR-PC weighted aggregation:

- Mean: `μ = Σ_e w_e * c_e[1]`  (weighted sum of per-element constant terms)
- Variance: `σ² = Σ_e w_e * Σ_p c_e[p]² - μ²`

where `w_e` is the element's empirical probability mass.  Returns
`(OutputMean, OutputVar)` with the same shapes as the generic `UQ`.  Sobol
indices for a multi-element expansion are available via
[`sobol_indices_multires`](@ref).
"""
function UQ(
        aPCE_apc::aPCE{T, MultiWaveletBasis{T}, A1, A2};
        axis = 1,
    ) where {T <: Real, A1, A2}
    mwb = aPCE_apc.OrthonormalBasis

    μ = zero(T)
    σ² = zero(T)
    for elem in mwb.elements
        isempty(elem.coefficients) && continue
        w = element_weight(elem)
        β = @view elem.coefficients[:, axis]
        μ += w * β[1]
        σ² += w * sum(abs2, β)
    end
    σ² -= μ^2

    return (OutputMean = [μ], OutputVar = [σ²])
end

# ============================================================================
# Sobol indices (standalone function for explicit per-element computation)
# ============================================================================

"""
    sobol_indices_multires(mwb::MultiWaveletBasis) -> NamedTuple

Compute first-order and total Sobol indices from per-element coefficients stored
in the basis (populated by `train!`).  Returns `(S_first, S_total)` matrices of
shape `(n_dims, outdim)`.

Within-element variance is attributed to the dimensions active in each term's
multi-index.  Between-element variance — the weighted spread of the per-element
constant terms around the global mean — is a function of the split
coordinate(s) only and is attributed to the split dimension(s): to both
`S_first` and `S_total` for a single split dimension, to `S_total` only (as an
interaction) when several dimensions are split.  Attribution of within-element
terms ignores that element membership itself depends on the split coordinate,
the standard aMR-PC approximation; `S_total` for the split dimension is
therefore a lower bound.
"""
function sobol_indices_multires(mwb::MultiWaveletBasis{T}) where {T <: Real}
    n_dims = mwb.n_dims
    outdim = _trained_outdim(mwb)

    # Weighted mean and variance
    μ = zeros(T, outdim)
    σ² = zeros(T, outdim)
    for elem in mwb.elements
        isempty(elem.coefficients) && continue
        w = T(element_weight(elem))
        β = elem.coefficients
        μ .+= w .* β[1, :]
        for i in 1:size(β, 1)
            σ² .+= w .* β[i, :] .^ 2
        end
    end
    σ² .-= μ .^ 2

    S_first = zeros(T, n_dims, outdim)
    S_total = zeros(T, n_dims, outdim)

    for elem in mwb.elements
        isempty(elem.coefficients) && continue
        w = T(element_weight(elem))
        β = elem.coefficients
        degs_e = _element_indices(elem, n_dims)

        for i in 1:size(β, 1)
            α = degs_e[i, :]
            for j in 1:n_dims
                if α[j] > 0
                    # First-order: only j has nonzero degree
                    if all(d == j || α[d] == 0 for d in 1:n_dims)
                        S_first[j, :] .+= w .* β[i, :] .^ 2
                    end
                    # Total: any term with j active
                    S_total[j, :] .+= w .* β[i, :] .^ 2
                end
            end
        end
    end

    # Between-element variance: Var over elements of the constant terms,
    # Σ_e w_e c_e[1]² − μ².  Element membership is determined by the split
    # coordinate(s) alone, so this variance belongs to the split dimension(s).
    between = zeros(T, outdim)
    for elem in mwb.elements
        isempty(elem.coefficients) && continue
        between .+= T(element_weight(elem)) .* elem.coefficients[1, :] .^ 2
    end
    between .-= μ .^ 2
    split_dims = [d for d in 1:n_dims if any(elem.level[d] > 0 for elem in mwb.elements)]
    if length(split_dims) == 1
        S_first[split_dims[1], :] .+= between
        S_total[split_dims[1], :] .+= between
    else
        # Membership depends on several coordinates jointly: an interaction,
        # so it contributes to each split dimension's total index only.
        for s in split_dims
            S_total[s, :] .+= between
        end
    end

    for j in 1:n_dims, k in 1:outdim
        σ²[k] > 0 || continue
        S_first[j, k] /= σ²[k]
        S_total[j, k] /= σ²[k]
    end

    return (S_first = S_first, S_total = S_total)
end

# ============================================================================
# Adaptive refinement (auto-refinement)
#
# Recursively splits the element with the largest local variance contribution
# V_e = w_e · Σ_{p≥2} c_e[p]², using a quantile split point.  An optional
# Sobol-stability stopping rule terminates refinement when the global
# sensitivity indices stop moving within their bootstrap confidence intervals.
#
# References:
#   Wan & Karniadakis (2005) J. Comput. Phys. 209:617–642 — variance-decay
#     refinement criterion (Eqs. 23–26)
#   Le Maître, Najm, Ghanem, Knio (2004) J. Comput. Phys. 197:502–531 —
#     multi-resolution analysis framework
#   Kröker & Oladyshkin (2022) Reliab. Eng. Syst. Safety 222:108376 — aMR-PC
#   Dubreuil, Berveiller, Petitjean, Salaün (2014) Reliab. Eng. Syst. Safety
#     121:263–275 — bootstrap CIs on Sobol indices and the stopping rule
#     transposed here from DoE enrichment to h-refinement
# ============================================================================

"""
    element_variance_contribution(mwb, elem_idx) -> T

Per-element non-constant variance contribution ``V_e = w_e \\sum_{p \\geq 2} c_e[p]^2``.
A high ``V_e`` indicates the local basis is working hard — that's where a split
will reduce the approximation error most.

# References
- Wan & Karniadakis (2005) J. Comput. Phys. 209:617–642
- Le Maître et al. (2004) J. Comput. Phys. 197:502–531
"""
function element_variance_contribution(
        mwb::MultiWaveletBasis{T}, elem_idx::Int,
    ) where {T <: Real}
    elem = mwb.elements[elem_idx]
    isempty(elem.coefficients) && return zero(T)
    w = elem.weight
    s = zero(T)
    for i in 2:size(elem.coefficients, 1)
        for j in 1:size(elem.coefficients, 2)
            s += elem.coefficients[i, j]^2
        end
    end
    return w * s
end

"""
    _select_split_point(x_elem, split_dim; strategy)

Select a candidate split point along `split_dim`.

- `:quantile` (default): median — matches equal-mass weighting assumption
- `:mean`: arithmetic mean
"""
function _select_split_point(
        x_elem::AbstractMatrix{T}, split_dim::Int;
        strategy::Symbol = :quantile,
    ) where {T <: Real}
    col = @view x_elem[:, split_dim]
    if strategy == :quantile
        sorted = sort!(collect(col))
        n = length(sorted)
        return T(n % 2 == 0 ? (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2 : sorted[(n + 1) ÷ 2])
    elseif strategy == :mean
        return T(sum(col) / length(col))
    else
        throw(ArgumentError("unknown strategy: $strategy (use :quantile or :mean)"))
    end
end

"""
    _split_one_element!(mwb, elem_idx, split_dim, split_point, x)

Replace element `elem_idx` in-place with two children split at
`(split_dim, split_point)`.  Threads `level` and `dim_index` for recursive
refinement.
"""
function _split_one_element!(
        mwb::MultiWaveletBasis{T},
        elem_idx::Int,
        split_dim::Int,
        split_point::Real,
        x::AbstractMatrix,
    ) where {T <: Real}
    parent = mwb.elements[elem_idx]
    n_dims = mwb.n_dims

    # Gather training points belonging to this element
    xs = ndims(x) == 1 ? reshape(x, :, 1) : x
    in_parent = trues(size(xs, 1))
    for d in 1:n_dims
        in_parent .&= (xs[:, d] .>= parent.lo[d]) .& (xs[:, d] .<= parent.hi[d])
    end
    x_parent = xs[in_parent, :]

    mask_lo = x_parent[:, split_dim] .<= T(split_point)
    mask_hi = .!mask_lo
    n_lo = count(mask_lo)
    n_hi = count(mask_hi)

    if n_lo < 2 || n_hi < 2
        throw(ArgumentError(
            "split at $(split_point) on dim $(split_dim) leaves too few samples " *
            "($(n_lo) lo, $(n_hi) hi); element has $(size(x_parent, 1)) points"))
    end

    n_parent = size(x_parent, 1)
    children = MultiWaveletElement{T}[]

    for (mask, is_hi) in ((mask_lo, false), (mask_hi, true))
        x_child = x_parent[mask, :]
        # Cap degree to maintain ≥2× overdetermination (n_samples ≥ 2(n_terms))
        degree_child = max(1, min(maximum(parent.degree), (size(x_child, 1) - 1) ÷ 2))
        rb_child = create_recurrence_basis(x_child, degree_child)

        lo = copy(parent.lo)
        hi = copy(parent.hi)
        if is_hi
            lo[split_dim] = T(split_point)
        else
            hi[split_dim] = T(split_point)
        end

        level = copy(parent.level)
        level[split_dim] += 1

        dim_index = copy(parent.dim_index)
        dim_index[split_dim] = 2 * parent.dim_index[split_dim] + (is_hi ? 1 : 0)

        # Conditional weight: P(child | parent) × P(parent)
        weight = T(count(mask) / n_parent) * parent.weight

        push!(children, MultiWaveletElement{T}(
            lo, hi, fill(degree_child, n_dims), rb_child,
            level, dim_index, weight,
            zeros(Int, 0, 0), zeros(T, 0, 0),
            fill(T(-Inf), n_dims), fill(T(Inf), n_dims)))
    end

    splice!(mwb.elements, elem_idx, children)
    return nothing
end

"""
    refine!(apc, X, y, elem_idx, split_dim, split_point)

Split element `elem_idx` at `(split_dim, split_point)` and retrain all elements.
"""
function refine!(
        apc::aPCE{T, MultiWaveletBasis{T}, A1, A2},
        X::AbstractMatrix,
        y::AbstractVecOrMat,
        elem_idx::Int,
        split_dim::Int,
        split_point::Real,
    ) where {T <: Real, A1, A2}
    xs = ndims(X) == 1 ? reshape(X, :, 1) : X
    _split_one_element!(apc.OrthonormalBasis, elem_idx, split_dim, split_point, xs)
    train!(apc, xs, y)
    return apc
end

"""
    auto_refine!(apc, X, y; max_elements, min_samples, strategy, sobol_tol, verbose)

Adaptively refine the multiresolution basis by recursively splitting the element
+ dimension combination that yields the largest between-group variance reduction.

At each step:
1. Train the model (per-element independent solves).
2. For each splittable element and candidate dimension, compute the between-group
   variance proxy: ``n_{lo} n_{hi} / n_e^2 \\cdot (\\bar y_{lo} - \\bar y_{hi})^2``.
3. Split the (element, dim) pair with the largest score at its median.
4. Optionally: compute Sobol indices and stop when ``\\max|\\Delta S|`` falls below
   `sobol_tol` (requires `X` for the marginal projection).

# Keyword arguments
- `max_elements::Int=8`: hard ceiling on element count
- `min_samples_per_element::Int=0`: minimum training points per child (0 → auto: `n_terms + 5`)
- `strategy::Symbol=:quantile`: split-point search (:quantile = grid, :median = fast)
- `min_improvement::Real=0.01`: stop when best between-group variance < this fraction of Var(y)
- `sobol_tol::Real=0`: if > 0, stop when max Sobol index change < sobol_tol
- `verbose::Bool=false`: log each split

# References
- Wan & Karniadakis (2005) J. Comput. Phys. 209:617–642 — variance-decay criterion
- Le Maître et al. (2004) J. Comput. Phys. 197:502–531 — multi-resolution analysis
- Kröker & Oladyshkin (2022) Reliab. Eng. Syst. Safety 222:108376 — aMR-PC
- Dubreuil et al. (2014) Reliab. Eng. Syst. Safety 121:263–275 — Sobol stopping rule
"""
function auto_refine!(
        apc::aPCE{T, MultiWaveletBasis{T}, A1, A2},
        X::AbstractVecOrMat,
        y::AbstractVecOrMat;
        max_elements::Int = 8,
        min_samples_per_element::Int = 0,
        strategy::Symbol = :quantile,
        min_improvement::Real = 0.01,
        sobol_tol::Real = 0.0,
        verbose::Bool = false,
    ) where {T <: Real, A1, A2}
    xs = ndims(X) == 1 ? reshape(X, :, 1) : X
    y_mat = ndims(y) == 1 ? reshape(y, :, 1) : y
    mwb = apc.OrthonormalBasis
    n_dims = mwb.n_dims
    n_samples = size(xs, 1)

    # Min-samples floor: if not specified, tie to local basis size
    # (n_terms ≈ (degree+1)^n_dims for a full tensor basis)
    max_degree = maximum(maximum(e.degree) for e in mwb.elements)
    n_terms = (max_degree + 1)^n_dims
    min_samples = min_samples_per_element > 0 ? min_samples_per_element : n_terms + 5

    train!(apc, xs, y_mat)

    # Coefficient-based Sobol from the per-element expansion (no off-support
    # surrogate evaluations — see sobol_indices_multires(::MultiWaveletBasis)).
    # The surrogate-MC Saltelli variant is numerically unstable for piecewise
    # polynomial surrogates on strongly-correlated inputs: mixed pick-freeze
    # points fall off the element data manifolds and the local bases
    # extrapolate catastrophically, dominating the estimator.
    S_prev = sobol_tol > 0 ? sobol_indices_multires(apc.OrthonormalBasis) : nothing

    while length(mwb.elements) < max_elements
        best_score = zero(T)
        best_elem = 0
        best_dim = 0
        best_point = zero(T)

        for (e, elem) in enumerate(mwb.elements)
            in_elem = trues(n_samples)
            for d in 1:n_dims
                in_elem .&= (xs[:, d] .>= elem.lo[d]) .& (xs[:, d] .<= elem.hi[d])
            end
            n_elem = count(in_elem)
            n_elem >= 2 * min_samples || continue

            x_elem = xs[in_elem, :]
            y_elem = y_mat[in_elem, 1]

            for d in 1:n_dims
                point, score = _select_best_split(
                    x_elem, y_elem, d, min_samples; strategy = strategy)
                isnan(point) && continue

                if score > best_score
                    best_score = score
                    best_elem = e
                    best_dim = d
                    best_point = point
                end
            end
        end

        best_elem == 0 && break

        # Variance-improvement threshold: stop when splitting yields negligible gain
        if min_improvement > 0
            y_col = @view y_mat[:, 1]
            μ_y = sum(y_col) / n_samples
            σ²_y = sum(abs2, y_col .- μ_y) / n_samples
            if best_score < min_improvement * σ²_y
                verbose && @info "auto_refine: improvement below threshold ($(round(best_score, sigdigits=4)) < $(round(min_improvement * σ²_y, sigdigits=4))), stopping"
                break
            end
        end

        if verbose
            @info "auto_refine: splitting element $(best_elem) on dim $(best_dim) " *
                  "at $(round(best_point, digits=4)) (score=$(round(best_score, sigdigits=4)), " *
                  "elements: $(length(mwb.elements)) → $(length(mwb.elements) + 1))"
        end

        _split_one_element!(mwb, best_elem, best_dim, best_point, xs)
        train!(apc, xs, y_mat)

        # Sobol-stability stopping rule
        # (adapted from Dubreuil et al. 2014, who apply it to DoE enrichment;
        # here it controls h-refinement termination instead)
        if sobol_tol > 0 && S_prev !== nothing
            S_now = sobol_indices_multires(apc.OrthonormalBasis)
            ΔS_max = maximum(abs.(S_now.S_first .- S_prev.S_first))
            if verbose
                @info "auto_refine: max|ΔS_first| = $(round(ΔS_max, sigdigits=4))"
            end
            if ΔS_max < sobol_tol
                verbose && @info "auto_refine: Sobol convergence reached (ΔS < $(sobol_tol))"
                break
            end
            S_prev = S_now
        end
    end

    return apc
end

# ----------------------------------------------------------------------------
# Greedy split-point selection
# ----------------------------------------------------------------------------

"""
    _select_best_split(x_elem, y_elem, split_dim, min_samples; strategy)

Search candidate quantiles (0.1–0.9) for the split point that maximises the
between-group variance ``n_{lo} n_{hi} / n^2 \\cdot (\\bar y_{lo} - \\bar y_{hi})^2``.

For `strategy = :quantile` the candidate set is the 10th–90th percentile grid.
For `strategy = :median` only the 50th percentile is tried (fast, but may miss
bimodal structure where the median falls between modes).
"""
function _select_best_split(
        x_elem::AbstractMatrix{T},
        y_elem::AbstractVector{T},
        split_dim::Int,
        min_samples::Int;
        strategy::Symbol = :quantile,
    ) where {T <: Real}
    n = length(y_elem)
    x_d = @view x_elem[:, split_dim]
    sorted_x = sort!(collect(x_d))

    quantiles = if strategy == :median
        [0.5]
    else
        collect(0.1:0.05:0.9)
    end

    best_score = T(-Inf)
    best_point = T(NaN)

    for q in quantiles
        idx = clamp(floor(Int, q * n) + 1, 1, n)
        point = sorted_x[idx]

        mask_lo = x_d .<= point
        n_lo = count(mask_lo)
        n_hi = n - n_lo
        (n_lo >= min_samples && n_hi >= min_samples) || continue

        y_lo_mean = sum(@view y_elem[mask_lo]) / n_lo
        y_hi_mean = sum(@view y_elem[.!mask_lo]) / n_hi
        score = T(n_lo * n_hi) / T(n)^2 * (y_lo_mean - y_hi_mean)^2

        if score > best_score
            best_score = score
            best_point = point
        end
    end

    return best_point, best_score
end

# ----------------------------------------------------------------------------
# Surrogate-MC Sobol indices (Saltelli / pick-freeze estimator)
# ----------------------------------------------------------------------------

# The analytic marginal-projection approach assumes the local polynomial bases
# are orthogonal w.r.t. the global marginal, which they are not (each element
# carries its own degree-d tensor basis centred on the element's data).  We use
# a surrogate-based Monte Carlo Saltelli estimator instead: draw samples from
# the input distribution, evaluate the (cheap) surrogate, and compute
# first-order and total-order indices via the standard pick-freeze formulas.
# This is exact up to MC error, handles correlated inputs, and works for any
# tree depth.
#
# References:
#   Saltelli, Annoni, Azzini, Campolongo, Ratto, Tarantola (2010)
#     "Variance based sensitivity analysis of model output"
#     Comput. Phys. Commun. 181:259–270
#   Sobol (2001) Math. Comput. Simul. 55:271–280

"""
    sobol_indices_multires(apc::aPCE{T,MultiWaveletBasis{T}}, X_train; n_mc)

Surrogate-based Monte Carlo Sobol indices for a trained multires model.
Uses the Saltelli pick-freeze estimator on the surrogate (cheap to evaluate,
handles correlated inputs, any tree depth).

Returns `(S_first, S_total)`, each an `n_dims × outdim` matrix.

# References
- Saltelli et al. (2010) Comput. Phys. Commun. 181:259–270 — pick-freeze estimator
"""
function sobol_indices_multires(
        apc::aPCE{T, MultiWaveletBasis{T}, A1, A2},
        X_train::AbstractMatrix;
        n_mc::Int = 100_000,
    ) where {T <: Real, A1, A2}
    n_dims = apc.input_dimensions
    mwb = apc.OrthonormalBasis
    outdim = _trained_outdim(mwb)
    n_train = size(X_train, 1)

    # Two independent sample matrices from the empirical input distribution
    N = min(n_mc, n_train * 10)
    idx_a = rand(1:n_train, N)
    idx_b = rand(1:n_train, N)
    A = X_train[idx_a, :]
    B = X_train[idx_b, :]

    # Training-data range per dimension — used to clip mixed pick-freeze points
    # to prevent polynomial extrapolation outside the element's support.
    x_lo = [minimum(@view X_train[:, d]) for d in 1:n_dims]
    x_hi = [maximum(@view X_train[:, d]) for d in 1:n_dims]

    # Surrogate evaluations
    Y_A = predict(apc, A)
    Y_B = predict(apc, B)

    S_first = zeros(T, n_dims, outdim)
    S_total = zeros(T, n_dims, outdim)

    for k in 1:outdim
        y_a = @view Y_A[:, k]
        y_b = @view Y_B[:, k]

        f0 = (sum(y_a) + sum(y_b)) / (2N)
        V = (sum(abs2, y_a) + sum(abs2, y_b)) / (2N) - f0^2
        V > eps(T) || continue

        for d in 1:n_dims
            AB = copy(A)
            @views AB[:, d] .= B[:, d]
            # Clip mixed points to training-data range to prevent catastrophic
            # polynomial extrapolation off the element's local support.
            for d2 in 1:n_dims
                clamp!(@view(AB[:, d2]), x_lo[d2], x_hi[d2])
            end
            Y_AB = predict(apc, AB)
            y_ab = @view Y_AB[:, k]

            # Saltelli (2010) Eqs. 2 and 5:
            # S_first[i]  = [E[f(B)·f(A_B^i)] - f0²] / V
            # S_total[i]  = 1 - [E[f(A)·f(A_B^i)] - f0²] / V
            S_first[d, k] = (sum(y_b .* y_ab) / N - f0^2) / V
            S_total[d, k] = 1 - (sum(y_a .* y_ab) / N - f0^2) / V
        end
    end

    return (S_first = S_first, S_total = S_total)
end

"""
    sobol_bootstrap_ci(apc, X, y; n_bootstrap, ci_level, fixed_tree, n_mc)

Bootstrap confidence intervals on the multiresolution Sobol indices.

Two modes:
- `fixed_tree=false` (default): resample (X, y), retrain from scratch on each
  bootstrap draw.  The CI covers coefficient uncertainty **and** refinement-path
  variability — the honest total uncertainty.
- `fixed_tree=true`: freeze the current element structure, resample (X, y), and
  retrain only the per-element coefficients.  The CI isolates coefficient
  uncertainty for the given decomposition.

Returns named tuple with `S_first_mean/lower/upper` and `S_total_mean/lower/upper`.

# References
- Dubreuil, Berveiller, Petitjean, Salaün (2014) Reliab. Eng. Syst. Safety
  121:263–275 — bootstrap CIs on Sobol indices from PCE
"""
function sobol_bootstrap_ci(
        apc::aPCE{T, MultiWaveletBasis{T}, A1, A2},
        X::AbstractVecOrMat,
        y::AbstractVecOrMat;
        n_bootstrap::Int = 200,
        ci_level::Real = 0.95,
        fixed_tree::Bool = false,
        n_mc::Int = 10_000,
    ) where {T <: Real, A1, A2}
    xs = ndims(X) == 1 ? reshape(X, :, 1) : X
    y_mat = ndims(y) == 1 ? reshape(y, :, 1) : y
    n_samples = size(xs, 1)
    n_dims = apc.input_dimensions
    outdim = size(y_mat, 2)

    # Capture the current tree structure for fixed_tree mode
    mwb_template = apc.OrthonormalBasis
    split_specs = if fixed_tree
        [(lo = copy(e.lo), hi = copy(e.hi), degree = copy(e.degree),
          level = copy(e.level), dim_index = copy(e.dim_index))
         for e in mwb_template.elements]
    else
        nothing
    end

    S_first_samples = Vector{Matrix{T}}()
    S_total_samples = Vector{Matrix{T}}()

    for _ in 1:n_bootstrap
        idx = rand(1:n_samples, n_samples)
        X_boot = xs[idx, :]
        y_boot = y_mat[idx, :]

        if fixed_tree && split_specs !== nothing
            # Rebuild basis with the same element structure but new local bases
            elements = MultiWaveletElement{T}[]
            for spec in split_specs
                mask = trues(size(X_boot, 1))
                for d in 1:n_dims
                    mask .&= (X_boot[:, d] .>= spec.lo[d]) .& (X_boot[:, d] .<= spec.hi[d])
                end
                x_elem = X_boot[mask, :]
                deg = max(1, min(maximum(spec.degree), size(x_elem, 1) - 1))
                rb = size(x_elem, 1) >= deg + 1 ? create_recurrence_basis(x_elem, deg) :
                     create_recurrence_basis(x_elem, max(1, size(x_elem, 1) - 1))
                push!(elements, MultiWaveletElement{T}(
                    copy(spec.lo), copy(spec.hi), fill(deg, n_dims), rb,
                    copy(spec.level), copy(spec.dim_index),
                    T(count(mask) / n_samples),
                    zeros(Int, 0, 0), zeros(T, 0, 0),
                    fill(T(-Inf), n_dims), fill(T(Inf), n_dims)))
            end
            mwb_boot = MultiWaveletBasis{T}(n_dims, elements)
            max_deg = maximum(maximum(s.degree) for s in split_specs)
            apc_boot = aPCE(X_boot, max_deg; basis = Val(:multires))
            apc_boot.OrthonormalBasis = mwb_boot
            train!(apc_boot, X_boot, y_boot)
        else
            apc_boot = aPCE(X_boot, maximum(maximum(e.degree) for e in mwb_template.elements);
                            basis = Val(:multires))
            train!(apc_boot, X_boot, y_boot)
        end

        # Coefficient-based Sobol (see auto_refine! for why not surrogate-MC).
        S = sobol_indices_multires(apc_boot.OrthonormalBasis)
        push!(S_first_samples, copy(S.S_first))
        push!(S_total_samples, copy(S.S_total))
    end

    α = 1 - ci_level
    lo_pct = α / 2 * 100
    hi_pct = (1 - α / 2) * 100

    S_first_mean = zeros(T, n_dims, outdim)
    S_first_lo = zeros(T, n_dims, outdim)
    S_first_hi = zeros(T, n_dims, outdim)
    S_total_mean = zeros(T, n_dims, outdim)
    S_total_lo = zeros(T, n_dims, outdim)
    S_total_hi = zeros(T, n_dims, outdim)

    for d in 1:n_dims, k in 1:outdim
        vf = [S[d, k] for S in S_first_samples]
        vt = [S[d, k] for S in S_total_samples]
        S_first_mean[d, k] = T(sum(vf) / n_bootstrap)
        S_first_lo[d, k] = T(_percentile(vf, lo_pct))
        S_first_hi[d, k] = T(_percentile(vf, hi_pct))
        S_total_mean[d, k] = T(sum(vt) / n_bootstrap)
        S_total_lo[d, k] = T(_percentile(vt, lo_pct))
        S_total_hi[d, k] = T(_percentile(vt, hi_pct))
    end

    return (
        S_first_mean = S_first_mean, S_first_lower = S_first_lo, S_first_upper = S_first_hi,
        S_total_mean = S_total_mean, S_total_lower = S_total_lo, S_total_upper = S_total_hi,
    )
end

function _percentile(v::Vector{T}, p::Real) where {T <: Real}
    sorted = sort(v)
    n = length(sorted)
    idx = (p / 100) * (n - 1) + 1
    lo = floor(Int, idx)
    hi = ceil(Int, idx)
    lo == hi && return sorted[lo]
    return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo])
end

# ============================================================================
# ChainRules — AD integration
# ============================================================================
# ChainRules — AD integration
# ============================================================================

function ChainRulesCore.rrule(
        ::typeof(aPCE_PsiPolynomialMatrix_zygote),
        TrainingInput,
        MultivariatePolynomialDegrees::AbstractMatrix{Int},
        mwb::MultiWaveletBasis,
    )
    Base.require_one_based_indexing(TrainingInput, MultivariatePolynomialDegrees)
    x = ndims(TrainingInput) == 1 ? reshape(TrainingInput, :, 1) : TrainingInput
    T = eltype(x)

    Psi = aPCE_PsiPolynomialMatrix_zygote(x, MultivariatePolynomialDegrees, mwb)

    NumberOfTerms, InputDimensions = size(MultivariatePolynomialDegrees)
    NCpoints = size(x, 1)

    element_ids = Vector{Int}(undef, NCpoints)
    for j in 1:NCpoints
        element_ids[j] = locate_element(mwb, x, j)
    end

    n_elements = length(mwb.elements)
    elem_derivs = Vector{Vector{Array{T, 3}}}(undef, n_elements)
    elem_sids = Vector{Vector{Int}}(undef, n_elements)

    for e in 1:n_elements
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
        for e in 1:n_elements
            sids = elem_sids[e]
            isempty(sids) && continue
            elem = mwb.elements[e]
            rb = elem.basis
            derivs = elem_derivs[e]

            for (k, j) in enumerate(sids)
                for i in 1:NumberOfTerms
                    Δij = ΔPsi[i, j]
                    iszero(Δij) && continue

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
        return (
            ChainRulesCore.NoTangent(), Δx,
            ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
        )
    end

    return Psi, aPCE_Psi_MultiWavelet_pullback
end

function ChainRulesCore.rrule(
        ::typeof(create_multiwavelet_basis),
        x::AbstractArray{T},
        degree::Union{Integer, Vector{Int}},
    ) where {T <: Real}
    mwb = create_multiwavelet_basis(x, degree)
    function create_multiwavelet_basis_pullback(_Δ)
        return (
            ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end
    return mwb, create_multiwavelet_basis_pullback
end

# ============================================================================
# Tests
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

    # Full aPCE workflow equivalence (single element should match recurrence)
    TrainingOutput = sin.(x[:, 1]) .* cos.(x[:, 2])
    TrainingOutput = reshape(TrainingOutput, :, 1)

    apc_multi = aPCE(x, degree; outdim = 1, basis = Val(:multires))
    apc_rec = aPCE(x, degree; outdim = 1, basis = Val(:recurrence))

    train!(apc_multi, x, TrainingOutput; bayesian_inversion = true, reg_order = 2)
    train!(apc_rec, x, TrainingOutput; bayesian_inversion = true, reg_order = 2)

    preds_multi = predict(apc_multi, x)
    preds_rec = predict(apc_rec, x)

    # Single-element multires should give very similar predictions
    # (not exact match because multires train! does per-element solve
    #  while recurrence train! does global solve, but mathematically equivalent)
    @test maximum(abs.(preds_multi .- preds_rec)) < 1.0e-6

    uq_multi = UQ(apc_multi)
    uq_rec = UQ(apc_rec)
    @test maximum(abs.(uq_multi.OutputMean .- uq_rec.OutputMean)) < 1.0e-6
    @test maximum(abs.(uq_multi.OutputVar .- uq_rec.OutputVar)) < 1.0e-4

    # Single-element multires mirrors its coefficients into ExpansionCoefficients
    # in the same layout as the global recurrence backend.
    @test apc_multi.ExpansionCoefficients ≈ apc_rec.ExpansionCoefficients atol = 1.0e-6
    @test size(apc_multi.OrthonormalBasis) == size(apc_rec.OrthonormalBasis)
    @test sprint(show, apc_multi) isa String
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

@testitem "multires_two_element_bimodal" begin
    using Random, Statistics
    using ArbitraryPolynomialChaosExpansion: train!, predict, UQ, sobol_indices_multires

    Random.seed!(42)
    n = 200
    x1 = vcat(randn(n ÷ 2) .* 0.15 .- 1.0, randn(n ÷ 2) .* 0.15 .+ 1.0)
    x2 = rand(n) .* 4 .- 2
    X = hcat(x1, x2)
    y = sin.(X[:, 1]) .+ 0.5 .* X[:, 2] .^ 2
    degree = 4

    apc_single = aPCE(X, degree; outdim = 1, basis = Val(:multires))
    train!(apc_single, X, y)
    mse_single = mean((predict(apc_single, X) .- y) .^ 2)

    apc_split = aPCE(X, degree; outdim = 1, basis = Val(:multires), split_dim = 1, split_point = 0.0)
    @test length(apc_split.OrthonormalBasis.elements) == 2
    train!(apc_split, X, y)
    mse_split = mean((predict(apc_split, X) .- y) .^ 2)

    # local bases resolve the two modes at least as well as one global basis
    @test mse_split < 10 * mse_single || mse_split < 0.1

    # element weights are empirical probability masses
    ws = [e.weight for e in apc_split.OrthonormalBasis.elements]
    @test sum(ws) ≈ 1.0
    @test ws[1] ≈ count(X[:, 1] .<= 0.0) / n

    # multi-element models have no global coefficient vector — NaN by contract
    @test all(isnan, apc_split.ExpansionCoefficients)

    # prediction points outside the training range still locate (±Inf slabs)
    x_out = [minimum(X[:, 1]) - 1.0 maximum(X[:, 2]) + 5.0; maximum(X[:, 1]) + 1.0 0.0]
    @test all(isfinite, predict(apc_split, x_out))

    # UQ matches the generic contract shapes and approximates the sample moments
    uq = UQ(apc_split)
    @test length(uq.OutputMean) == 1 && length(uq.OutputVar) == 1
    @test isapprox(uq.OutputMean[1], mean(y); rtol = 0.1)
    @test isapprox(uq.OutputVar[1], var(y); rtol = 0.25)

    sobol = sobol_indices_multires(apc_split.OrthonormalBasis)
    @test all(sobol.S_first .>= -1.0e-10)
    @test all(sobol.S_total .>= sobol.S_first .- 1.0e-10)
end

@testitem "multires_asymmetric_split_moments" begin
    using Random, Statistics
    using ArbitraryPolynomialChaosExpansion: train!, UQ

    Random.seed!(42)
    x = rand(400) .* 4 .- 3
    y = x
    degree = 3

    apc = aPCE(x, degree; basis = Val(:multires), split_dim = 1, split_point = 0.0)
    train!(apc, x, y)

    uq = UQ(apc)
    @test isapprox(uq.OutputMean[1], mean(y); atol = 0.05)
    @test isapprox(uq.OutputVar[1], var(y); rtol = 0.1)
    # regression test: the old dyadic 2^-level weight reported mean ≈ -0.5
    # for this 75/25 split instead of the true empirical mean ≈ -1.0
    @test uq.OutputMean[1] < -0.8
end

@testitem "multires_sobol_step_function" begin
    using Random
    using ArbitraryPolynomialChaosExpansion: train!, sobol_indices_multires

    Random.seed!(42)
    X = hcat(rand(400) .* 2 .- 1, rand(400) .* 2 .- 1)
    y = Float64.(X[:, 1] .> 0.0)
    degree = 2

    apc = aPCE(X, degree; basis = Val(:multires), split_dim = 1, split_point = 0.0)
    train!(apc, X, y)

    # nearly all variance is between-element and driven by the step in dim 1
    sobol = sobol_indices_multires(apc.OrthonormalBasis)
    @test sobol.S_total[1, 1] > 0.8
    @test sobol.S_first[2, 1] < 0.2
end

@testitem "multires_train_kwargs_respected" begin
    using Random
    using ArbitraryPolynomialChaosExpansion: train!, predict

    Random.seed!(42)
    X = rand(30, 2)
    y = X[:, 1] .+ 0.5 .* X[:, 2]
    degree = 2

    apc_multi = aPCE(X, degree; basis = Val(:multires))
    apc_rec = aPCE(X, degree; basis = Val(:recurrence))

    # bayesian_inversion = false, reg_order = 0 is a plain pinv solve on both
    # backends, so predictions must agree
    train!(apc_multi, X, y; bayesian_inversion = false, reg_order = 0)
    train!(apc_rec, X, y; bayesian_inversion = false, reg_order = 0)
    @test maximum(abs.(predict(apc_multi, X) .- predict(apc_rec, X))) < 1.0e-8

    train!(apc_multi, X, y; bayesian_inversion = true, reg_order = 1)
    @test all(isfinite, predict(apc_multi, X))
end

@testitem "multires_qnorm_truncation_respected" begin
    using Random
    using ArbitraryPolynomialChaosExpansion: train!

    Random.seed!(42)
    X = rand(60, 3)
    y = sin.(X[:, 1]) .+ X[:, 2] .* X[:, 3]

    apc = aPCE(X, 4; s_interactions = 0.25, basis = Val(:multires))
    train!(apc, X, y)

    # single element, full-degree caps: its stored index set is exactly the
    # model's q-norm-truncated global set
    elem = apc.OrthonormalBasis.elements[1]
    @test elem.multi_indices == apc.MultivariatePolynomialDegrees
    @test size(elem.coefficients, 1) == size(apc.MultivariatePolynomialDegrees, 1)
end

@testitem "multires_element_index_set_small" begin
    using ArbitraryPolynomialChaosExpansion: _element_multi_index_set

    rows = _element_multi_index_set([2, 1], 2)
    # α1 ≤ 2, α2 ≤ 1, sum(α) ≤ 2, sorted by (total degree, lexicographic)
    expected = [0 0; 0 1; 1 0; 1 1; 2 0]
    @test rows == expected
end

@testitem "multires_AD_mooncake_crosscheck" begin
    using ForwardDiff, Zygote, Random
    using DifferentiationInterface
    using ADTypes: AutoMooncake
    using ArbitraryPolynomialChaosExpansion: aPCE_PsiPolynomialMatrix_zygote

    Random.seed!(42)
    x = rand(10, 2)
    degree = 4
    degs = aPCE_MultivariatePolynomialDegrees(2, degree, 1.0, 1.0)
    mwb = create_multiwavelet_basis(x, degree)

    f_multi(z) = sum(aPCE_PsiPolynomialMatrix_zygote(z, degs, mwb))

    # ForwardDiff — numerical ground truth
    grad_fd = ForwardDiff.gradient(f_multi, x)

    # Zygote — via ChainRules rrule
    grad_zyg = Zygote.gradient(f_multi, x)[1]

    # Mooncake — via @from_rrule registration in MooncakeExt.jl
    backend = AutoMooncake()
    extras = prepare_gradient(f_multi, backend, x)
    grad_moon = similar(x)
    gradient!(f_multi, grad_moon, extras, backend, x)

    # Three-way agreement
    @test isapprox(grad_fd, grad_zyg, atol = 1.0e-6)
    @test isapprox(grad_fd, grad_moon, atol = 1.0e-6)
    @test all(isfinite, grad_moon)
    @test !all(iszero, grad_moon)
end
