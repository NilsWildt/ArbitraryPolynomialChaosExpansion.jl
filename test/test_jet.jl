# JET static analysis (re-enabled).
#
# JET was previously disabled (the old file was a commented-out Pluto notebook)
# under the belief it was incompatible with Julia 1.12. JET 0.11+ works on
# 1.12; the earlier breakage was a stale cached build.
#
# We gate `report_opt` (optimization / dynamic-dispatch analysis) on the hot
# numerical kernels and the core Psi-matrix assembly, which are verified clean
# (0 reports). The high-level `predict`/`train!` paths still contain a few
# dynamic dispatches through heavy third-party deps and are intentionally not
# asserted clean here (see Stage 5).

@testitem "JET_optimization_clean_on_core_kernels" begin
    using JET
    M = ArbitraryPolynomialChaosExpansion

    targets = [
        (M.evalpoly_two, (Float64, Vector{Float64})),
        (M.evaluate_polynomial_horner_array, (Vector{Float64}, Vector{Float64})),
        (M.compute_moments!, (Vector{Float64}, Vector{Float64}, Int, Int)),
        (M.aPCE_PsiPolynomialMatrix, (Matrix{Float64}, Matrix{Int64}, Array{Float64, 3})),
    ]
    for (f, types) in targets
        report = JET.report_opt(f, types)
        @test isempty(JET.get_reports(report))
    end
end
