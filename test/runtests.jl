using Pkg: Pkg
using Chairmarks
using DrWatson
using ForwardDiff
# using JET
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

# DIT_PATH = joinpath(@__DIR__, "..", "..", "ArbitraryPolynomialChaosExpansion.jl")
# if isdir(DIT_PATH)
#     Pkg.develop(; path=DIT_PATH)
# else
#     Pkg.add("ArbitraryPolynomialChaosExpansion")
# end


# Run all @testitem tests from the package
TestItemRunner.@run_package_tests
