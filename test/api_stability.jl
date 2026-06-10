# API-stability locks.
#
# This package has downstream dependents, so the *public* surface must not
# change without an intentional, reviewed update to the snapshot below.
# These tests fail loudly on any accidental export addition or removal.
#
# Ground truth captured from `names(ArbitraryPolynomialChaosExpansion)` on the
# v0.2.4 baseline (21 exported symbols). Note that `APCEGradientOverrides` is a
# *submodule*; its exports stay in the submodule namespace and are deliberately
# NOT part of the top-level public API.

@testitem "public_API_surface_lock" begin
    expected = Set([
        :CenteredBasis,
        :GaussianCollocation,
        :PsiPolynomialMatrix_zygote,
        :UQ,
        :aPCE,
        :aPCE_FullBasis,
        :aPCE_MultivariatePolynomialDegrees,
        :aPCE_OrthonormalBasis,
        :aPCE_PsiPolynomialMatrix,
        :compose_Ψ,
        :compute_moments!,
        :create_basis,
        :create_centered_basis,
        :evaluate_Ψ,
        :normalization_functions,
        :partitionTrainTest,
        :predict,
        :predict_from_coeffs,
        :reverse_columns!,
        :special_sort_two_arrays!,
        :train!,
    ])
    actual = Set(setdiff(
        names(ArbitraryPolynomialChaosExpansion),
        [:ArbitraryPolynomialChaosExpansion],
    ))

    removed = setdiff(expected, actual)   # would break downstream `using` code
    added = setdiff(actual, expected)     # unintended new public surface

    @test isempty(removed)
    @test isempty(added)
    @test actual == expected
    @test length(actual) == 21
end

@testitem "public_API_callables_defined" begin
    M = ArbitraryPolynomialChaosExpansion
    callables = [
        :aPCE, :train!, :predict, :predict_from_coeffs, :UQ, :GaussianCollocation,
        :create_basis, :create_centered_basis, :aPCE_PsiPolynomialMatrix,
        :aPCE_OrthonormalBasis, :aPCE_FullBasis, :aPCE_MultivariatePolynomialDegrees,
        :compose_Ψ, :evaluate_Ψ, :normalization_functions, :partitionTrainTest,
        :special_sort_two_arrays!, :reverse_columns!, :compute_moments!,
    ]
    for sym in callables
        @test isdefined(M, sym)
        @test !isempty(methods(getfield(M, sym)))
    end
    # CenteredBasis is a public type used by create_centered_basis.
    @test isdefined(M, :CenteredBasis)
    @test CenteredBasis isa Type
end

@testitem "public_API_undefined_exports_lock" begin
    # `PsiPolynomialMatrix_zygote` is exported (so it shows up in `names`) but
    # was never bound: the implementation is the unexported
    # `aPCE_PsiPolynomialMatrix_zygote`. The export name is missing the `aPCE_`
    # prefix, so any downstream `using` + call hits an UndefVarError. This test
    # locks that known wart and fails if any *new* undefined export appears.
    # See Stage 2 for the alias fix that makes the public name functional.
    M = ArbitraryPolynomialChaosExpansion
    undefined_exports = [
        s for s in names(M)
            if s != :ArbitraryPolynomialChaosExpansion && !isdefined(M, s)
    ]
    @test Set(undefined_exports) == Set([:PsiPolynomialMatrix_zygote])
    @test isdefined(M, :aPCE_PsiPolynomialMatrix_zygote)  # the real implementation
end

@testitem "public_API_core_signatures_lock" begin
    # Structural signature locks for the most-depended-on entry points, so a
    # silent change to accepted argument types is caught.
    @test hasmethod(aPCE, Tuple{Matrix{Float64}, Int64})
    @test hasmethod(aPCE, Tuple{Vector{Float64}, Int64})        # AbstractVecOrMat
    @test hasmethod(predict, Tuple{aPCE{Float64}, Matrix{Float64}})
    @test hasmethod(UQ, Tuple{aPCE{Float64}})
    @test hasmethod(train!, Tuple{aPCE{Float64}, Matrix{Float64}, Matrix{Float64}})
end
