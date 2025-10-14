using Revise
using DrWatson

@quickactivate "ArbitraryPolynomialChaosExpansion"
module Runner
    using CairoMakie
    using Statistics
    using DrWatson
    using PrettyTables
    using Logging
    using TerminalLoggers: TerminalLogger
    using ProgressLogging
    using Logging
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

    function run()
        err = []
        ts = []
        ps = []

        FT = Float64

        @timeit to "load_data" begin
            #  Loading Input distributions, Training Data and Validation Data
            # Extracting the variables
            # @info size(file["Xtr"])
            # n = 1000
            # m = 23
            m_out = 1
            rng = Xoshiro(42)
            file = matread(datadir("Data.mat"))
            indall = 1:1000
            @info keys(file)
            TrainingInput = file["TrainingInput"][indall, :] .|> FT
            Input_distribution = file["Input_distributions"][:, :] .|> FT
            TrainingOutput = file["TrainingOutput"][indall, :] .|> FT
            ValidationInput = file["ValidationInput"][indall, :] .|> FT
            ValidationOutput = file["ValidationOutput"][indall, :] .|> FT
        end

        degree = 8
        s_marginals = FT(1.0)
        s_interactions = FT(1.0)
        @info "" size(TrainingOutput, 2)
        apc_instance = @timeit to "aPC_instance" APCE.aPCE(Input_distribution, degree; outdim = size(TrainingOutput, 2), is_orthonormal = true, s_marginals = s_marginals, s_interactions = s_interactions, normalize_data = true, do_gauss = false)
        @info "" apc_instance.NumberOfTerms
        @assert apc_instance.NumberOfTerms < 5000 "Too many coefficients: $(apc_instance.NumberOfTerms)"

        # TrainingInput_selection = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy=:PCM)
        # @info "" TrainingInput_selection
        # # Find indices of TrainingInput elements that are in TrainingInput_selection
        # # selected_indices = findall(in(TrainingInput_selection), TrainingInput)
        # # # # Select rows from TrainingInput based on the found indices
        # # TrainingOutput_selection = TrainingInput[selected_indices, :]
        # # Find indices where elements in TrainingInput match elements in TrainingInput_selection
        #     selected_indices = Int[]

        #     # Loop through each row of TrainingInput and check if it matches any row in TrainingInput_selection
        #     for i in 1:size(TrainingInput, 1)
        #         row = TrainingInput[i, :]
        #         if any(r -> r == row, eachrow(TrainingInput_selection))
        #             push!(selected_indices, i)
        #         end
        #     end

        #     # Select rows from TrainingInput based on the found indices
        #     TrainingOutput_selection = TrainingInput[selected_indices, :]

        #     # Log the sizes of the selection and original data for debugging
        #     @info "Training Input Selection size: $(size(TrainingInput_selection))"
        #     @info "Training Output Selection size: $(size(TrainingOutput_selection))"
        #     @info "Original Training Input size: $(size(TrainingInput))"
        #     @info "Original Training Output size: $(size(TrainingOutput))"
        @warn "Not using gaussian root training points."

        # TrainingInput = @timeit to "KMeansCollocation" KMeansCollocation(apc_instance)
        @timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_order = 3)
        PredictionOutput = @timeit to "prediction" predict(apc_instance, TrainingInput)
        ValidationPredictionOutput = @timeit to "prediction" predict(apc_instance, ValidationInput)

        uq = UQ(apc_instance)
        # data = ["Mean" uq.OutputMean[1] mean(ValidationOutput); "Var" uq.OutputVar[1] var(ValidationOutput)]
        data = ["Mean" mean(ValidationPredictionOutput) mean(ValidationOutput); "Var" var(ValidationPredictionOutput) var(ValidationOutput)]

        headers = ["Type", "aPCE", "Data"]
        # Display the table
        pretty_table(data; header = headers)

        gratio = (1.0 + sqrt(5.0)) / 2.0
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
        fig = Makie.Figure(; size = (fig_width, fig_height))

        # Visualization of Training and Validation Performance
        plot_index = 1 # Track the plot index across both rows and columns
        for i in 1:size(TrainingOutput, 2)
            row, col = divrem(plot_index - 1, Int(num_columns)) .+ (1, 1)
            # Plot training performance
            ax1 = CairoMakie.Axis(fig[row, col], title = "Training Performance $i", xlabel = "Training Response", ylabel = "Prediction Response")
            CairoMakie.scatter!(ax1, TrainingOutput[:, i], PredictionOutput[:, i], color = :red, marker = :circle)
            plot_index += 1

            row, col = divrem(plot_index - 1, Int(num_columns)) .+ (1, 1)
            # Plot validation performance
            ax2 = CairoMakie.Axis(fig[row, col], title = "Validation Performance $i", xlabel = "Validation Reference", ylabel = "Validation Response")
            CairoMakie.scatter!(ax2, ValidationOutput[:, i], ValidationPredictionOutput[:, i], color = :blue, marker = :circle)
            plot_index += 1
        end

        fig_path = normpath(plotsdir("ishigami_test1"))
        @info fig_path
        mkpath(fig_path)
        save(joinpath(fig_path, "training_validation_performance_$(degree)_$(s_marginals)_$(s_interactions).png"), fig)
        display(to)
        display(fig)

        return fig
    end
    run() |> display
end
