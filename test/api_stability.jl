# API-stability locks.
#
# This package has downstream dependents, so the *public* surface must not
# change without an intentional, reviewed update to the snapshot below.
# These tests fail loudly on any accidental export addition or removal.
#
# Ground truth captured from `names(ArbitraryPolynomialChaosExpansion)` on the
# v0.2.4 baseline (21 exported symbols), extended with the opt-in recurrence
# basis (`create_recurrence_basis`, `RecurrenceCenteredBasis`) → 23, and with
# the analytic derivative basis (`aPCE_DerivativeBasis`) → 24, and with the
# multiresolution basis (`MultiWaveletBasis`, `MultiWaveletElement`,
# `create_multiwavelet_basis`) → 27.  Note that `APCEGradientOverrides` is a
# *submodule*; its exports stay in the submodule namespace and are deliberately
# NOT part of the top-level public API.

@testitem "public_API_surface_lock" begin
    expected = Set(
        [
            :CenteredBasis,
            :GaussianCollocation,
            :PsiPolynomialMatrix_zygote,
            :RecurrenceCenteredBasis,
            :UQ,
            :aPCE,
            :aPCE_DerivativeBasis,
            :aPCE_FullBasis,
            :aPCE_MultivariatePolynomialDegrees,
            :aPCE_OrthonormalBasis,
            :aPCE_PsiPolynomialMatrix,
            :compose_Ψ,
            :compute_moments!,
            :create_basis,
            :create_centered_basis,
            :create_recurrence_basis,
            :create_multiwavelet_basis,
            :evaluate_Ψ,
            :normalization_functions,
            :partitionTrainTest,
            :predict,
            :predict_from_coeffs,
            :reverse_columns!,
            :special_sort_two_arrays!,
            :train!,
            :MultiWaveletBasis,
            :MultiWaveletElement,
        ]
    )
    actual = Set(
        setdiff(
            names(ArbitraryPolynomialChaosExpansion),
            [:ArbitraryPolynomialChaosExpansion],
        )
    )

    removed = setdiff(expected, actual)   # would break downstream `using` code
    added = setdiff(actual, expected)     # unintended new public surface

    @test isempty(removed)
    @test isempty(added)
    @test actual == expected
    @test length(actual) == 27
end

@testitem "public_API_callables_defined" begin
    M = ArbitraryPolynomialChaosExpansion
    callables = [
        :aPCE, :train!, :predict, :predict_from_coeffs, :UQ, :GaussianCollocation,
        :create_basis, :create_centered_basis, :aPCE_PsiPolynomialMatrix,
        :aPCE_DerivativeBasis,
        :aPCE_OrthonormalBasis, :aPCE_FullBasis, :aPCE_MultivariatePolynomialDegrees,
        :compose_Ψ, :evaluate_Ψ, :normalization_functions, :partitionTrainTest,
        :special_sort_two_arrays!, :reverse_columns!, :compute_moments!,
        :PsiPolynomialMatrix_zygote, :create_recurrence_basis,
        :create_multiwavelet_basis,
    ]
    for sym in callables
        @test isdefined(M, sym)
        @test !isempty(methods(getfield(M, sym)))
    end
    # CenteredBasis / RecurrenceCenteredBasis / MultiWaveletBasis are public types.
    @test isdefined(M, :CenteredBasis)
    @test CenteredBasis isa Type
    @test isdefined(M, :RecurrenceCenteredBasis)
    @test RecurrenceCenteredBasis isa Type
    @test isdefined(M, :MultiWaveletBasis)
    @test MultiWaveletBasis isa Type
    @test isdefined(M, :MultiWaveletElement)
    @test MultiWaveletElement isa Type
end

@testitem "public_API_no_undefined_exports" begin
    # Every exported symbol must be bound. `PsiPolynomialMatrix_zygote` used to
    # be an unbound export (the implementation is `aPCE_PsiPolynomialMatrix_zygote`,
    # which the export name omitted the `aPCE_` prefix of); it is now aliased so
    # the public name resolves. This guards against any new undefined export.
    M = ArbitraryPolynomialChaosExpansion
    undefined_exports = [
        s for s in names(M)
            if s != :ArbitraryPolynomialChaosExpansion && !isdefined(M, s)
    ]
    @test isempty(undefined_exports)
    # The public alias resolves to the internal implementation.
    @test PsiPolynomialMatrix_zygote === M.aPCE_PsiPolynomialMatrix_zygote
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
