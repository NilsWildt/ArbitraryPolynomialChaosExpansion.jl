using Pkg: Pkg
using Chairmarks
using ForwardDiff
using MethodAnalysis
using Preferences
using Random
using Test
using Aqua
using Supposition
using Revise
using Suppressor
using BenchmarkTools
using TestItemRunner
using ErrorTypes
using ComponentArrays
using Statistics
using LinearAlgebra
using FiniteDifferences
using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion

set_preferences!(ArbitraryPolynomialChaosExpansion, "precompile_workload" => true; force = true)

# Run all @testitem tests from the package
TestItemRunner.@run_package_tests
