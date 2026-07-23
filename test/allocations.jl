# Allocation regression guards for the hot numerical kernels.
#
# These catch accidental allocation regressions (e.g. a type instability or a
# stray temporary introduced in the scalar evaluation path). Bounds are tight
# where the kernel is genuinely non-allocating and generous elsewhere so the
# tests are not flaky across Julia versions / BLAS backends.
#
# Each measurement goes through a function barrier (`alloc_of`) so the arguments
# are locals, not @testitem-module globals -- otherwise @allocated would also
# count the boxing of non-const global accesses rather than just the call.

@testitem "alloc_evalpoly_two_is_zero" begin
    M = ArbitraryPolynomialChaosExpansion
    alloc_of(f, args...) = @allocated f(args...)
    coeffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = 2.0
    alloc_of(M.evalpoly_two, x, coeffs)             # warm up / compile
    @test alloc_of(M.evalpoly_two, x, coeffs) == 0
end

@testitem "alloc_horner_array_scales_with_output" begin
    M = ArbitraryPolynomialChaosExpansion
    alloc_of(f, args...) = @allocated f(args...)
    x = collect(1.0:50.0)
    coeffs = [1.0, 2.0, 3.0]
    alloc_of(M.evaluate_polynomial_horner_array, x, coeffs)   # warm up
    # The reversed-coefficient temporary is now hoisted out of the per-point
    # loop, so allocation is the output vector plus a single small temporary.
    @test alloc_of(M.evaluate_polynomial_horner_array, x, coeffs) <= 4 * length(x) * sizeof(Float64)
end

@testitem "alloc_PsiPolynomialMatrix_bounded" begin
    using Random
    M = ArbitraryPolynomialChaosExpansion
    alloc_of(f, args...) = @allocated f(args...)
    X = rand(Xoshiro(1), 20, 2)
    apc = aPCE(X, 2; outdim = 1, basis = Val(:monomial))
    mpd = apc.MultivariatePolynomialDegrees
    basis = apc.OrthonormalBasis
    alloc_of(M.aPCE_PsiPolynomialMatrix, X, mpd, basis)       # warm up
    # Generous upper bound: guards against gross regressions (the monomial kernel
    # uses scalar Horner evaluation to avoid per-element temporaries). The default
    # basis is now :auto (recurrence); this lock pins the monomial path explicitly.
    @test alloc_of(M.aPCE_PsiPolynomialMatrix, X, mpd, basis) <= 65_536
end
