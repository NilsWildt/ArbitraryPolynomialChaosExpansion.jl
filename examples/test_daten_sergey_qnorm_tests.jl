using Revise
using DrWatson
@quickactivate "ArbitraryPolynomialChaosExpansion"
module Runner
    using DrWatson
    using PrettyTables
    using PropDicts
    using Logging
    using TerminalLoggers: TerminalLogger
    using ProgressLogging
    using Logging
    using BenchmarkTools
    using Makie
    using CairoMakie
    using MAT
    using Random
    loggingdir(args...) = projectdir("output", "logs", args...)
    mkpath(loggingdir())
    # io = open(loggingdir("debug.txt"), "w")
    # file_logger = ConsoleLogger(io, Logging.Debug)
    debug_logging = ConsoleLogger(stderr, Logging.Debug)
    info_logging = ConsoleLogger(stderr, Logging.Info)
    # Here you may include files from the source directory
    global_logger(info_logging)
    include(srcdir("ArbitraryPolynomialChaosExpansion.jl"))
    using .ArbitraryPolynomialChaosExpansion
    const APCE = ArbitraryPolynomialChaosExpansion
    # using Preferences
    # set_preferences!(APCE, "precompile_workload" => false; force=true)

    using TimerOutputs
    const to = TimerOutput()

    function partitionTrainTest(data; at = 0.7, rng = Xoshiro())
        num_samples = size(data, 1)
        shuffled_indices = shuffle(rng, 1:num_samples)
        split_index = floor(Int, at * num_samples)

        train_indices = view(shuffled_indices, 1:split_index)
        test_indices = view(shuffled_indices, (split_index + 1):num_samples)

        X_train = data[train_indices]
        X_test = data[test_indices]

        return X_train, X_test
    end

    function run()
        err = []
        ts = []
        ps = []
        # degress = 1:1:5

        FT = Float64
        @timeit to "lod_data" begin
            #  Loading Input distributions, Training Data and Validation Data
            file = matread(datadir("Data.mat"))
            print(keys(file))
            # Extracting the variables
            # Input_distributions = file["Xtr"] |> Array{FT}
            # @info size(file["Xtr"])
            indall = 1:1000
            rng = Xoshiro(42)
            @info "" file["TrainingOutput"]
            itrain, itest = partitionTrainTest(indall; at = 0.75, rng)
            TrainingInput = file["TrainingInput"][indall, :] |> Array{FT}
            TrainingOutput = file["TrainingOutput"][indall, :] |> Array{FT}
            @info "" size(TrainingInput)

            ValidationInput = file["ValidationInput"][indall, :] |> Array{FT}
            ValidationOutput = file["ValidationOutput"][indall, :] |> Array{FT}
        end
        degree = 3
        B = create_basis(TrainingInput, degree; normalize_data = false)
        display(B)
        @info "" size(TrainingOutput, 2)
        apc_instance = @timeit to "aPC_instance" APCE.aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), OrthonormalRepresentation = true, qnorm = 0.6, normalize_data = true)
        @assert apc_instance.NumberOfTerms < 2000 "Too many coefficients."
        return apc_instance
        # TrainingInput = KMeansCollocation(apc_instance)
        # @debug "" TrainingInput
        # display(	@report_call GaussianCollocation(apc_instance; strategy = :PCM) )
        # @descend GaussianCollocation(apc_instance; strategy = :PCM)
        # TrainingInput = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy = :PCM)
        # TrainingInput = @timeit to "KMeansCollocation" KMeansCollocation(apc_instance)
        # @info "" size(TrainingInput)
        # @timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_mode=3)
        # PredictionOutput = @timeit to "prediction" predict(apc_instance, TrainingInput)
        # ValidationPredictionOutput = @timeit to "prediction" predict(apc_instance, ValidationInput)
    end
    run()
end
