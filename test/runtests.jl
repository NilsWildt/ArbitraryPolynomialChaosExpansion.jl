using Pkg: Pkg
using Preferences: set_preferences!
using TestItemRunner

# NOTE: the AD-backend packages below are loaded at top level on purpose, not
# as dead imports. TestItemRunner runs @testitem blocks in this process, and the
# Mooncake AD tests use `AutoMooncake()` / `AutoForwardDiff()` via
# DifferentiationInterface without importing the backend package themselves.
# DifferentiationInterface only activates a backend once its package is loaded in
# the process (via a package extension), so loading them here is what makes
# `check_available(AutoMooncake())` true inside those testitems. Removing them
# breaks the AD tests.
using ForwardDiff
using FiniteDifferences
using Mooncake

using ArbitraryPolynomialChaosExpansion

set_preferences!(ArbitraryPolynomialChaosExpansion, "precompile_workload" => true; force = true)

# Discover and run all @testitem tests in the package (src/ and test/).
TestItemRunner.@run_package_tests
