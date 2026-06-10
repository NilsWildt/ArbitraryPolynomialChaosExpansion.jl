# Coverage for previously-untested public exports:
# compose_Ψ, evaluate_Ψ, special_sort_two_arrays!, normalization_functions,
# partitionTrainTest (+ compute_moments! value check).
# Each test pins behavior and type stability (@inferred) where applicable.

@testitem "normalization_functions_roundtrip_and_inference" begin
    using Random, Statistics
    rng = Xoshiro(1)
    x = rand(rng, 10, 3)
    normalize, inverse_normalize = normalization_functions(x)

    z = normalize(x)
    @test isapprox(inverse_normalize(z), x; rtol = 1.0e-12)
    @test all(abs.(vec(mean(z; dims = 1))) .< 1.0e-12)   # columns centered
    @test all(isapprox.(vec(std(z; dims = 1)), 1.0; rtol = 1.0e-12))  # unit std
    @test typeof(normalize(x)) == typeof(x)
    @test typeof(inverse_normalize(x)) == typeof(x)
    @inferred normalization_functions(x)
end

@testitem "partitionTrainTest_split_and_determinism" begin
    using Random
    data = collect(1.0:100.0)
    train, test = partitionTrainTest(data; at = 0.7, rng = Xoshiro(42))

    @test length(train) == 70
    @test length(test) == 30
    @test sort(vcat(train, test)) == data          # disjoint + complete
    @test isempty(intersect(train, test))

    # Same seed -> identical split (determinism).
    train2, test2 = partitionTrainTest(data; at = 0.7, rng = Xoshiro(42))
    @test train == train2
    @test test == test2

    @inferred partitionTrainTest(data; at = 0.7, rng = Xoshiro(42))
end

@testitem "special_sort_two_arrays!_orders_by_first_two_columns" begin
    # y is reordered to follow the lexicographic sort of x's first two columns.
    x = [3.0 1.0; 1.0 2.0; 2.0 0.0]
    y = [10.0, 20.0, 30.0]
    result = special_sort_two_arrays!(x, y)
    # Sort key on column 1: rows ordered (1.0, 2.0, 3.0) -> original indices 2,3,1
    @test result == [20.0, 30.0, 10.0]
    @inferred special_sort_two_arrays!([3.0 1.0; 1.0 2.0; 2.0 0.0], [10.0, 20.0, 30.0])
end

@testitem "compose_Ψ_shape_and_consistency" begin
    using Random
    rng = Xoshiro(99)
    X = rand(rng, 12, 2)
    y = rand(rng, 12, 1)
    apc = aPCE(X, 2; outdim = 1)
    train!(apc, X, y; bayesian_inversion = false)

    Psi = compose_Ψ(X, apc.MultivariatePolynomialDegrees, apc.OrthonormalBasis, apc.ExpansionDegree)
    @test size(Psi) == (size(X, 1), apc.NumberOfTerms)   # (points, terms)
    @test all(isfinite, Psi)
    @inferred compose_Ψ(X, apc.MultivariatePolynomialDegrees, apc.OrthonormalBasis, apc.ExpansionDegree)
end

@testitem "evaluate_Ψ_matches_predict" begin
    using Random
    rng = Xoshiro(99)
    X = rand(rng, 12, 2)
    y = rand(rng, 12, 1)
    apc = aPCE(X, 2; outdim = 1)
    train!(apc, X, y; bayesian_inversion = false)

    out = evaluate_Ψ(
        X, apc.ExpansionCoefficients, apc.MultivariatePolynomialDegrees,
        apc.OrthonormalBasis, apc.ExpansionDegree, :name,
    )
    @test isapprox(out, predict(apc, X); rtol = 1.0e-10)
    @inferred evaluate_Ψ(
        X, apc.ExpansionCoefficients, apc.MultivariatePolynomialDegrees,
        apc.OrthonormalBasis, apc.ExpansionDegree, :name,
    )
end

@testitem "compute_moments!_matches_definition" begin
    using Random
    rng = Xoshiro(5)
    data = rand(rng, 50)
    dd = 2
    m = zeros(2 * dd + 2)
    compute_moments!(m, data, length(data), dd)
    manual = [sum(data .^ i) / length(data) for i in 0:(2 * dd + 1)]
    @test isapprox(m, manual; rtol = 1.0e-12)
    @test m[1] == 1.0   # zeroth moment
    @inferred compute_moments!(zeros(2 * dd + 2), data, length(data), dd)
end
