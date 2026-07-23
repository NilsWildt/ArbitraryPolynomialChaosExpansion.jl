# Strengthened coverage for partially-tested public exports, including edge
# cases (degree 0), multi-dimensional inputs, and Float32 genericity to confirm
# the code is not silently over-constrained to Float64.

@testitem "aPCE_FullBasis_structure_and_genericity" begin
    using Random
    # FullBasis is the lower-triangular all-ones (monomial / Vandermonde) basis.
    for deg in 0:3
        fb = aPCE_FullBasis(rand(Xoshiro(deg), 10), deg)
        @test size(fb) == (deg + 1, deg + 1)
        @test fb == [i >= j ? 1.0 : 0.0 for i in 1:(deg + 1), j in 1:(deg + 1)]
    end
    # Float32 input -> Float32 basis (genericity, not hard-coded Float64).
    fb32 = aPCE_FullBasis(rand(Xoshiro(1), Float32, 10), 2)
    @test eltype(fb32) == Float32
    @inferred aPCE_FullBasis(rand(10), 2)
end

@testitem "aPCE_MultivariatePolynomialDegrees_total_degree_constraint" begin
    for (dims, deg) in [(1, 3), (2, 2), (3, 2)]
        mpd = aPCE_MultivariatePolynomialDegrees(dims, deg, 1.0, 1.0)
        @test size(mpd, 2) == dims
        @test maximum(sum(mpd; dims = 2)) <= deg     # total-degree truncation
        @test all(mpd .>= 0)
        @test eltype(mpd) <: Integer
    end
    @inferred aPCE_MultivariatePolynomialDegrees(2, 2, 1.0, 1.0)
end

@testitem "aPCE_OrthonormalBasis_orthonormality_and_genericity" begin
    using Random, LinearAlgebra
    # The basis must be orthonormal with respect to the empirical data measure.
    data = rand(Xoshiro(11), 500)
    deg = 3
    B = aPCE_OrthonormalBasis(data, deg)
    @test size(B) == (deg + 1, deg + 1)
    # P[i, k+1] = value of basis polynomial k at data point i.
    P = [
        sum(B[k + 1, j] * data[i]^(j - 1) for j in 1:(deg + 1))
            for i in eachindex(data), k in 0:deg
    ]
    gram = (P' * P) ./ length(data)
    @test isapprox(gram, I(deg + 1); atol = 1.0e-8)

    # Degree-0 edge case.
    B0 = aPCE_OrthonormalBasis(rand(Xoshiro(3), 50), 0)
    @test size(B0) == (1, 1)

    # Float32 genericity.
    B32 = aPCE_OrthonormalBasis(rand(Xoshiro(3), Float32, 100), 2)
    @test eltype(B32) == Float32
end

@testitem "create_centered_basis_orthonormal_on_skewed_data" begin
    using Random, LinearAlgebra, Statistics
    const APCE = ArbitraryPolynomialChaosExpansion
    # Regression for the empirical-normalization fix. The old moment-sum Hankel
    # norm suffered catastrophic cancellation, giving orthonormality defects up
    # to ~1e+26 already at degree 16. The basis must now stay orthonormal
    # w.r.t. the empirical measure even for strongly skewed marginals.
    #
    # Two tiers: a TIGHT bound over the degrees this package is actually used at
    # (≤ 12), and a LOOSE guard at very high degree. The loose tier only checks
    # the fix has not regressed to catastrophic cancellation — a sub-1e-6 defect
    # is unattainable there for *any* monomial-coefficient basis (verified: a
    # QR-Vandermonde coefficient basis is worse), because a degree ≳ 16
    # orthonormal polynomial on a skewed measure loses precision when its
    # monomial form is re-evaluated via `evalpoly`.
    rng = MersenneTwister(1)
    n = 5_000
    datasets = (
        "Exp(1)" => -log.(rand(rng, n)),
        "LogNormal" => exp.(randn(rng, n)),
    )
    defect(x, d) = let cb = APCE.create_centered_basis(x, d),
            mpd = APCE.aPCE_MultivariatePolynomialDegrees(1, d, 1.0, 1.0)
        Psi = Matrix(APCE.aPCE_PsiPolynomialMatrix_zygote(reshape(x, :, 1), mpd, cb)')
        opnorm(Psi' * Psi ./ n - I)
    end
    for (_name, x) in datasets
        for d in (1, 2, 4, 6, 8, 10, 12)          # tight: orthonormal to ~1e-6
            @test defect(x, d) < 1.0e-6
        end
        for d in (16, 20)                          # loose: no catastrophic blow-up
            @test defect(x, d) < 1.0
        end
    end
end

@testitem "create_recurrence_basis_high_degree_orthonormality" begin
    using Random, LinearAlgebra
    const APCE = ArbitraryPolynomialChaosExpansion
    # The opt-in three-term-recurrence basis evaluates orthonormal polynomials
    # via the forward recurrence instead of monomial coefficients + evalpoly, so
    # it keeps every value O(1) and stays orthonormal to much higher degree than
    # the monomial `CenteredBasis` (which the previous test caps at ~d=12).
    rng = MersenneTwister(1)
    n = 5_000
    x = -log.(rand(rng, n))                        # Exp(1), skewed

    defect(basis, d) = let mpd = APCE.aPCE_MultivariatePolynomialDegrees(1, d, 1.0, 1.0)
        Psi = Matrix(APCE.aPCE_PsiPolynomialMatrix_zygote(reshape(x, :, 1), mpd, basis)')
        opnorm(Psi' * Psi ./ n - I)
    end

    # Orthonormal to ~1e-6 well past the monomial ceiling.
    for d in (1, 2, 4, 8, 12, 16)
        @test defect(APCE.create_recurrence_basis(x, d), d) < 1.0e-6
    end
    @test defect(APCE.create_recurrence_basis(x, 20), 20) < 1.0e-4   # vs ~0.16 monomial

    # Strictly better than the monomial basis where the latter degrades.
    for d in (16, 20, 24)
        rb = APCE.create_recurrence_basis(x, d)
        cb = APCE.create_centered_basis(x, d)
        @test defect(rb, d) < defect(cb, d)
    end

    # Agrees with the monomial basis at low degree (orthonormal polys are unique
    # up to per-column sign), including at NEW evaluation points.
    d = 6
    mpd = APCE.aPCE_MultivariatePolynomialDegrees(1, d, 1.0, 1.0)
    Pc = Matrix(APCE.aPCE_PsiPolynomialMatrix_zygote(reshape(x, :, 1), mpd, APCE.create_centered_basis(x, d))')
    Pr = Matrix(APCE.aPCE_PsiPolynomialMatrix_zygote(reshape(x, :, 1), mpd, APCE.create_recurrence_basis(x, d))')
    @test isapprox(abs.(Pc), abs.(Pr); atol = 1.0e-8)

    xnew = reshape(collect(range(0.01, 5.0; length = 7)), :, 1)
    mpd4 = APCE.aPCE_MultivariatePolynomialDegrees(1, 4, 1.0, 1.0)
    Pr_new = APCE.aPCE_PsiPolynomialMatrix_zygote(xnew, mpd4, APCE.create_recurrence_basis(x, 4))
    Pc_new = APCE.aPCE_PsiPolynomialMatrix_zygote(xnew, mpd4, APCE.create_centered_basis(x, 4))
    @test isapprox(abs.(Pr_new), abs.(Pc_new); atol = 1.0e-8)

    # eltype genericity + multi-dimensional construction.
    rb32 = APCE.create_recurrence_basis(rand(Xoshiro(2), Float32, 200, 2), 3)
    @test eltype(rb32) == Float32
    @test size(rb32.α) == (3, 2)
    @test size(rb32.β) == (4, 2)
end

@testitem "create_centered_basis_closed_form_consistent_across_d4_d5_boundary" begin
    using Random, LinearAlgebra
    const APCE = ArbitraryPolynomialChaosExpansion
    # Regression for the degree ≤ 4 closed-form path: it used to return *monic*
    # (un-normalized) polynomials, so d ≤ 4 bases were not orthonormal and were
    # inconsistent with the d ≥ 5 Stieltjes path at the boundary. Now every
    # degree routes through the normalized Stieltjes recurrence.
    rng = MersenneTwister(1)
    n = 5_000
    x = -log.(rand(rng, n))                       # Exp(1), skewed

    # (a) low-degree bases must themselves be orthonormal.
    for d in 1:4
        cb = APCE.create_centered_basis(x, d)
        mpd = APCE.aPCE_MultivariatePolynomialDegrees(1, d, 1.0, 1.0)
        Psi = Matrix(APCE.aPCE_PsiPolynomialMatrix_zygote(reshape(x, :, 1), mpd, cb)')
        @test opnorm(Psi' * Psi ./ n - I) < 1.0e-6
    end

    # (b) the d = 4 and d = 5 bases must agree on their shared degrees (0..4).
    cb4 = APCE.create_centered_basis(x, 4)
    cb5 = APCE.create_centered_basis(x, 5)
    mpd4 = APCE.aPCE_MultivariatePolynomialDegrees(1, 4, 1.0, 1.0)
    mpd5 = APCE.aPCE_MultivariatePolynomialDegrees(1, 5, 1.0, 1.0)
    Psi4 = Matrix(APCE.aPCE_PsiPolynomialMatrix_zygote(reshape(x, :, 1), mpd4, cb4)')
    Psi5 = Matrix(APCE.aPCE_PsiPolynomialMatrix_zygote(reshape(x, :, 1), mpd5, cb5)')
    # 1-D total-degree terms are ordered by degree, so the first 5 columns are
    # the degree-0..4 polynomials in both bases.
    @test isapprox(Psi4[:, 1:5], Psi5[:, 1:5]; atol = 1.0e-8)
end

@testitem "aPCE_PsiPolynomialMatrix_3arg_shape_and_genericity" begin
    using Random
    rng = Xoshiro(5)
    X = rand(rng, 12, 2)
    apc = aPCE(X, 2; outdim = 1)
    Psi = aPCE_PsiPolynomialMatrix(X, apc.MultivariatePolynomialDegrees, apc.OrthonormalBasis)
    @test size(Psi) == (apc.NumberOfTerms, size(X, 1))   # (terms, points)
    @test all(isfinite, Psi)

    # Float32 genericity.
    X32 = rand(Xoshiro(5), Float32, 12, 2)
    apc32 = aPCE(X32, 2; outdim = 1)
    Psi32 = aPCE_PsiPolynomialMatrix(X32, apc32.MultivariatePolynomialDegrees, apc32.OrthonormalBasis)
    @test eltype(Psi32) == Float32
end

@testitem "PsiPolynomialMatrix_zygote_public_alias_works" begin
    using Random
    # `PsiPolynomialMatrix_zygote` is the public export aliasing the internal
    # `aPCE_PsiPolynomialMatrix_zygote`. It must be callable (was previously an
    # unbound export) and agree with the internal implementation.
    rng = Xoshiro(7)
    X = rand(rng, 10, 2)
    apc = aPCE(X, 2; outdim = 1)
    via_public = PsiPolynomialMatrix_zygote(X, apc.MultivariatePolynomialDegrees, apc.OrthonormalBasis)
    via_internal = ArbitraryPolynomialChaosExpansion.aPCE_PsiPolynomialMatrix_zygote(
        X, apc.MultivariatePolynomialDegrees, apc.OrthonormalBasis,
    )
    @test via_public == via_internal
    @test size(via_public) == (apc.NumberOfTerms, size(X, 1))
end
