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
    P = [sum(B[k + 1, j] * data[i]^(j - 1) for j in 1:(deg + 1))
        for i in eachindex(data), k in 0:deg]
    gram = (P' * P) ./ length(data)
    @test isapprox(gram, I(deg + 1); atol = 1.0e-8)

    # Degree-0 edge case.
    B0 = aPCE_OrthonormalBasis(rand(Xoshiro(3), 50), 0)
    @test size(B0) == (1, 1)

    # Float32 genericity.
    B32 = aPCE_OrthonormalBasis(rand(Xoshiro(3), Float32, 100), 2)
    @test eltype(B32) == Float32
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
