# Aqua code-quality checks.
#
# `using Aqua` was present in the old runtests.jl but `Aqua.test_all` was never
# actually called. This testitem wires it in.
#
# All checks are enabled. MKL is excluded from the stale-deps check because it
# is loaded conditionally (`using MKL` only on Intel x86_64); on other platforms
# it is never loaded, so Aqua would report it as stale even though it is a
# legitimate platform-specific BLAS backend.

@testitem "Aqua_quality" begin
    using Aqua
    Aqua.test_all(
        ArbitraryPolynomialChaosExpansion;
        stale_deps = (ignore = [:MKL],),
    )
end
