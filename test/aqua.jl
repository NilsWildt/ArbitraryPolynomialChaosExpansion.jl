# Aqua code-quality checks.
#
# `using Aqua` was present in the old runtests.jl but `Aqua.test_all` was never
# actually called. This testitem wires it in.
#
# deps_compat and stale_deps are disabled here and tightened in Stage 3 (they
# require the [compat] bounds and dependency pruning done in that stage). The
# remaining checks -- undefined exports, method ambiguities, type piracy,
# project-extras consistency, unbound type parameters -- pass as of Stage 2.

@testitem "Aqua_quality" begin
    using Aqua
    Aqua.test_all(
        ArbitraryPolynomialChaosExpansion;
        deps_compat = false,   # Stage 3: add [compat] bounds, then enable
        stale_deps = false,    # Stage 3: prune unused deps, then enable
    )
end
