using Revise
using DrWatson
@quickactivate "ArbitraryPolynomialChaosExpansion"
using ShareAdd
@usingany Makie
@usingany CairoMakie
@usingany UnPack
module Runner
    using DrWatson
    using PrettyTables
    using PropDicts
    using Logging
    using TerminalLoggers: TerminalLogger
    using ProgressLogging
    using Logging
    # using BenchmarkTools
    using StaticArrays

    using UnPack
    using MAT
    using Random
    using Distributions
    using Makie
    using CairoMakie
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

    """
    # Returns
    - `X_train`: The training set features.
    - `X_test`: The testing set features.

    The function randomly shuffles the dataset and splits it according to the specified training proportion (`at`).
    """
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
    using Distributions
    #  Base.similar(::Vector{Union{}}) = undef
    function testfunction(N, d_out; seed = 123)
        tvec = 1.0
        if d_out > 1
            tvec = LinRange(0, 1, d_out)
        end
        function partitionTrainTest(data; at = 0.7, rng = Xoshiro(seed))
            num_samples = size(data, 1)
            shuffled_indices = shuffle(rng, 1:num_samples)
            split_index = floor(Int, at * num_samples)
            train_indices = view(shuffled_indices, 1:split_index)
            test_indices = view(shuffled_indices, (split_index + 1):num_samples)
            X_train = data[train_indices]
            X_test = data[test_indices]
            return X_train, X_test
        end


        rng = Xoshiro(seed)
        # D_din ==2
        x = [rand(rng, Beta(2, 1), N) rand(rng, Beta(3, 1), N) rand(rng, Normal(0, 1), N)]
        # ModelResponse = reduce(hcat, [@. (x[:, 1] .^ 2 + x[:, 2] - 1.0) .^ 2 .+ x[:, 1]^3 + 0.5 * x[:, 1] * exp(x[:, 2]) .- sqrt.(t) .* x[:, 1]  for t in tvec])
        # for i ∈ 3:lastindex(x, 1)
        #     ModelResponse .+= x[i]
        # end

        a = 7.0
        b = 0.1
        ModelResponse = [(sin.(x[:, 1]) .+ a .* (sin.(x[:, 2]) .^ 2) .+ b .* x[:, 3] .^ 4 .* sin.(x[:, 1])) .* sqrt(t) for t in tvec]
        ModelResponse = reduce(hcat, ModelResponse)
        # @info "" size(x) size(ModelResponse)
        Data = hcat(x, ModelResponse)
        idxlist = 1:size(x, 1) |> collect
        Trainind, Testind = partitionTrainTest(idxlist)
        return (TrainingInput = Data[Trainind, 1:2], TrainingOutput = Data[Trainind, 3:end], ValidationInput = Data[Testind, 1:2], ValidationOutput = Data[Testind, 3:end])
    end

    function run()
        rng = Xoshiro(123)
        FT = Float64
        m_out = 5
        @unpack TrainingInput, TrainingOutput, ValidationInput, ValidationOutput = testfunction(1000, m_out)
        @info "" size(TrainingInput) size(TrainingOutput) size(ValidationInput) size(ValidationOutput)

        degree = 23
        s_marginals = FT(1.0)
        s_interactions = FT(1.0)
        @info "" size(TrainingOutput, 2)
        apc_instance = @timeit to "aPC_instance" APCE.aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), OrthonormalRepresentation = true, s_marginals = s_marginals, s_interactions = s_interactions, normalize_data = true)
        @warn "" apc_instance.NumberOfTerms apc_instance.NumberOfTerms / m_out
        @assert apc_instance.NumberOfTerms < 5000 "Too many coefficients: $(apc_instance.NumberOfTerms)"
        # TrainingInput = KMeansCollocation(apc_instance)
        # @debug "" TrainingInput
        # display(	@report_call GaussianCollocation(apc_instance; strategy = :PCM) )
        # @descend GaussianCollocation(apc_instance; strategy = :PCM)
        # TrainingInput = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy = :PCM)
        # TrainingInput = @timeit to "KMeansCollocation" KMeansCollocation(apc_instance)
        @timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_order = 3)
        PredictionOutput = @timeit to "prediction" predict(apc_instance, TrainingInput)
        ValidationPredictionOutput = @timeit to "prediction" predict(apc_instance, ValidationInput)
        # Back to regular matrices:
        # TrainingOutput = Matrix(TrainingOutput)
        # ValidationOutput = Matrix(ValidationOutput)
        # PredictionOutput = Matrix(PredictionOutput)
        uq = UQ(apc_instance)
        data = ["Mean" uq.OutputMean[1] mean(ValidationOutput); "Var" uq.OutputVar[1] var(ValidationOutput)]
        headers = ["Type", "aPCE", "Data"]
        # Display the table
        pretty_table(data; column_labels = headers)


        gratio = (1.0 + sqrt(5.0)) / 2.0
        # Assuming TrainingOutput, PredictionOutput, ValidationOutput, and ValidationPredictionOutput are defined
        num_plots = size(TrainingOutput, 2) * 2 # Total number of plots (training + validation for each column)

        # Aim for a square layout
        num_columns = ceil(sqrt(num_plots / gratio))
        num_rows = ceil(num_plots / num_columns)

        # Set dimensions for each subplot
        subplot_width = 300 # Width in pixels for each subplot
        subplot_height = subplot_width / gratio # Height determined by golden ratio

        # Calculate total figure dimensions
        fig_width = subplot_width * num_columns
        fig_height = subplot_height * num_rows
        fig = Figure(size = (fig_width, fig_height))

        # Visualization of Training and Validation Performance
        plot_index = 1 # Track the plot index across both rows and columns
        for i in 1:size(TrainingOutput, 2)
            row, col = divrem(plot_index - 1, Int(num_columns)) .+ (1, 1)
            # Plot training performance
            ax1 = Axis(fig[row, col], title = "Training Performance $i", xlabel = "Training Response", ylabel = "Prediction Response")
            scatter!(ax1, TrainingOutput[:, i], PredictionOutput[:, i], color = :red, marker = :circle)
            plot_index += 1

            row, col = divrem(plot_index - 1, Int(num_columns)) .+ (1, 1)
            # Plot validation performance
            ax2 = Axis(fig[row, col], title = "Validation Performance $i", xlabel = "Validation Reference", ylabel = "Validation Response")
            scatter!(ax2, ValidationOutput[:, i], ValidationPredictionOutput[:, i], color = :blue, marker = :circle)
            plot_index += 1
        end
        fig_path = normpath(plotsdir("testfunction"))
        @info fig_path
        mkpath(fig_path)
        save(joinpath(fig_path, "training_validation_performance_$(degree)_$(s_marginals)_$(s_interactions).png"), fig)

        display(fig)

        display(to)
        return fig
    end
    run()
end
