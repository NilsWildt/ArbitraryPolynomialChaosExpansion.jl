# Golden-value regression locks for the end-to-end workflow.
#
# These pin the *observable behavior* (trained coefficients, predictions, UQ)
# of the full aPCE -> train! -> predict -> UQ pipeline against fixed-seed
# inputs. They are the parity oracle for the Stage-4 solver refactor: the
# try/catch fallback ladder in `train!` is being replaced with condition-based
# logic, and these tests guarantee the numerical results do not move.
#
# Reference values were captured on the v0.2.4 baseline (pre-refactor).
# Tolerances: pinv-only paths are pure linear algebra (tight rtol); Bayesian
# paths run an iterative regularized solver (looser rtol to absorb
# platform-level FP noise while still catching any real behavioral change).
#
# These locks pin the *monomial-coefficient* basis path explicitly. The default
# `aPCE(...)` basis moved to `:auto` (recurrence for the orthonormal case), so
# each call requests `basis = Val(:monomial)` to keep exercising the exact
# v0.2.4 code path these baselines were captured on (the coefficient values are
# basis-dependent; predictions are basis-invariant).

@testitem "golden_A_deg1_2D_1out_bayes" begin
    using Random
    rng = Xoshiro(42)
    X = rand(rng, 10, 2)
    y = rand(rng, 10, 1)
    apc = aPCE(X, 1; outdim = 1, center_data = true, basis = Val(:monomial))
    train!(apc, X, y; bayesian_inversion = true, reg_order = 2)
    pred = predict(apc, X)
    uq = UQ(apc)

    ref_coeffs = [0.7855451185710457; 0.10852610995657774; -0.8007469530812737;;]
    ref_pred = [
        0.7056762808770755; 0.5750665794048906; 0.6014073849200607;
        0.6721767340751378; 0.401253191033304; 0.46634009316013153;
        0.5510765777694093; 0.18314478282253377; 0.003482334383618557;
        0.7185392222618783;;
    ]

    @test isapprox(apc.ExpansionCoefficients, ref_coeffs; rtol = 1.0e-6)
    @test isapprox(pred, ref_pred; rtol = 1.0e-6)
    @test isapprox(uq.OutputMean, [0.7855451185710457]; rtol = 1.0e-6)
    @test isapprox(uq.OutputVar, [0.6529735994112508]; rtol = 1.0e-6)
end

@testitem "golden_B_deg2_2D_2out_bayes" begin
    using Random
    rng = Xoshiro(123)
    X = rand(rng, 20, 2)
    y = rand(rng, 20, 2)
    apc = aPCE(X, 2; outdim = 2, center_data = true, basis = Val(:monomial))
    train!(apc, X, y; bayesian_inversion = true, reg_order = 2)
    pred = predict(apc, X)
    uq = UQ(apc)

    ref_coeffs = [
        2.473588197261801 2.0361531497864873;
        -0.9325346747323011 -1.2420182046881039;
        0.28822409907737345 -0.3597596995763863;
        1.5891534076716047 1.2009917074385112;
        -0.40801660259769434 0.7954715912674918;
        0.16812268677484232 -0.015844783653168035
    ]

    @test isapprox(apc.ExpansionCoefficients, ref_coeffs; rtol = 1.0e-6)
    @test size(pred) == (20, 2)
    @test isapprox(pred[1, :], [0.7767208624510233, 0.7003013928912991]; rtol = 1.0e-6)
    @test isapprox(pred[end, :], [0.5637964773140682, 0.5905461251218791]; rtol = 1.0e-6)
    @test isapprox(uq.OutputMean, [2.473588197261801]; rtol = 1.0e-6)
    @test isapprox(uq.OutputVar, [3.6728453897850724]; rtol = 1.0e-6)
end

@testitem "golden_C_deg2_pinv_only" begin
    using Random
    # No Bayesian inversion -> pure pinv path. Pin tightly.
    rng = Xoshiro(7)
    X = rand(rng, 15, 2)
    y = rand(rng, 15, 1)
    apc = aPCE(X, 2; outdim = 1, basis = Val(:monomial))
    train!(apc, X, y; bayesian_inversion = false)
    pred = predict(apc, X)

    ref_coeffs = [
        -1.0131865270476204; 2.2361426726088793; -0.9242778017840025;
        -2.384452013308292; 0.22722147860850475; 1.3415170837507602;;
    ]
    ref_pred = [
        0.4814510925484208; 0.5784528346327067; 0.7012022164932881;
        0.584620760824651; 0.6483101797903361; 0.4313547151133108;
        0.675944475469823; 0.5144925858369676; 0.31383610724879873;
        0.6094094660185814; 0.5367438842539556; 0.3373218884414422;
        0.4251331889568788; 0.4823232784857776; -0.025392317102235218;;
    ]

    @test isapprox(apc.ExpansionCoefficients, ref_coeffs; rtol = 1.0e-9)
    @test isapprox(pred, ref_pred; rtol = 1.0e-9)
end

@testitem "golden_D_rank_deficient_input" begin
    using Random
    # Perfectly correlated input columns -> ill-conditioned Psi. This is the
    # case that stresses the solver-robustness fallback ladder in train!.
    rng = Xoshiro(2024)
    base = rand(rng, 12, 1)
    X = hcat(base, base)
    y = rand(rng, 12, 1)
    apc = aPCE(X, 2; outdim = 1, basis = Val(:monomial))
    train!(apc, X, y; bayesian_inversion = true, reg_order = 1)
    pred = predict(apc, X)

    ref_pred = [
        0.4074356641336784; 0.657251632880491; 0.5758685026935618;
        0.6577200364320077; 0.46259459072031517; 0.6356156424735506;
        0.3408340387607168; 0.541086535713571; 0.47486887363893215;
        0.6146259609135349; 0.3914636839475203; 0.4955070991470787;;
    ]

    @test all(isfinite, pred)
    @test all(isfinite, apc.ExpansionCoefficients)
    @test isapprox(pred, ref_pred; rtol = 1.0e-6)
end
