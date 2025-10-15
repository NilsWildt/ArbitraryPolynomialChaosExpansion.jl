### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 4e32423b-8a3a-45d6-b00c-276c7936dc5a
import Pkg

# ╔═╡ 685184f1-3d61-40d9-a30f-82a17d1e48c5
begin
    using Revise
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
    using MethodAnalysis
    using JET
    loggingdir(args...) = projectdir("output", "logs", args...)
    mkpath(loggingdir())
    # io = open(loggingdir("debug.txt"), "w")
    # file_logger = ConsoleLogger(io, Logging.Debug)
    debug_logging = ConsoleLogger(stderr, Logging.Debug)
    info_logging = ConsoleLogger(stderr, Logging.Info)
    # Here you may include files from the source directory
    global_logger(info_logging)
    include(srcdir("APCE.jl"))
    using .APCE
    import .APCE: train!
    import .APCE: predict
    import .APCE: UQ

    using Test

    # using Preferences
    # set_preferences!(APCE, "precompile_workload" => false; force=true)
    using TimerOutputs

end

# ╔═╡ e390926a-3b59-4019-90b3-578c493d7867
Pkg.activate(normpath(raw"\\iws-ls3-cifs.tik.uni-stuttgart.de\shared\users\ac125867\03_projects\32_aPC_julia\APCE.jl"))

# ╔═╡ f7847c56-f124-4716-b38c-4582752807d6
html"""<style>
main {
    max-width: 96%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 5cd351d0-5635-486d-b890-61bfc7af7c46
report_file(normpath(raw"\\iws-ls3-cifs.tik.uni-stuttgart.de\shared\users\ac125867\03_projects\32_aPC_julia\APCE.jl\examples\test_maria_data.jl"))

# ╔═╡ 22c1871f-0a7f-425e-9194-ac6a26de6f4c
report_package("APCE")

# ╔═╡ 2a3366e0-cf21-4ea6-a626-4789dd1ab002
begin
    mis = methodinstances(APCE)    # get all the compiled methodinstances for functions owned by the package
    # Now let's filter out the ones that pass without issue
    badmis = filter(mis) do mi
        return !isempty(JET.get_reports(report_call(mi)))
        # JET.get_reports(report_call(mi))
    end
    badmis
end

# ╔═╡ 534e7802-63f3-43d5-99d6-2f25cd33b3ba
# JET.report_file(normpath(raw"\\iws-ls3-cifs.tik.uni-stuttgart.de\shared\users\ac125867\03_projects\32_aPC_julia\APCE.jl\examples\test_maria_data.jl"))

# ╔═╡ 63931b1e-fba0-49e8-a19f-9102ac5f144b
const to = TimerOutput()

# ╔═╡ 829cb50a-90c0-4017-9533-2c28e11c7213
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

# ╔═╡ cf0f8cfd-69d6-4b68-bdfd-8a8ca4dadc61
function run()
    err = []
    ts = []
    ps = []
    # degress = 1:1:5

    FT = Float64
    @timeit to "lod_data" begin
        #  Loading Input distributions, Training Data and Validation Data
        file = matread(datadir("gw_training_data.mat"))
        print(keys(file))
        # Extracting the variables
        # Input_distributions = file["Xtr"] |> Array{FT}
        # @info size(file["Xtr"])
        indall = 1:100
        rng = Xoshiro(42)
        @info ""  file["TrainingOutput"]
        itrain, itest = partitionTrainTest(indall; at = 0.75, rng)
        TrainingInput = file["TrainingInput"][indall, :] |> Array{FT}
        TrainingOutput = file["TrainingOutput"][indall, :] |> Array{FT}
        @info "" size(TrainingInput)

        ValidationInput = file["ValidationInput"][indall, :] |> Array{FT}
        ValidationOutput = file["ValidationOutput"][indall, :] |> Array{FT}
    end
    degree = 2
    @info "" size(TrainingOutput, 2)
    apc_instance = @timeit to "aPC_instance" APCE.aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), OrthonormalRepresentation = true, qnorm = 0.6, normalize_data = true)
    @assert apc_instance.NumberOfTerms < 2000 "Too many coefficients."

    # TrainingInput = KMeansCollocation(apc_instance)
    # @debug "" TrainingInput
    # display(	@report_call GaussianCollocation(apc_instance; strategy = :PCM) )
    # @descend GaussianCollocation(apc_instance; strategy = :PCM)
    # TrainingInput = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy = :PCM)
    # TrainingInput = @timeit to "KMeansCollocation" KMeansCollocation(apc_instance)
    @info "" size(TrainingInput)
    @timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_mode = 3)


    PredictionOutput = @timeit to "prediction" predict(apc_instance, TrainingInput)
    ValidationPredictionOutput = @timeit to "prediction" predict(apc_instance, ValidationInput)


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
    fig_path = normpath(plotsdir("maria_test2"))
    @info fig_path
    mkpath(fig_path)
    # save(joinpath(fig_path,"training_validation_performance.png"), fig)

    display(fig)

    display(to)
    return fig
end

# ╔═╡ 2d55378b-b7cb-49b8-8222-422aa0d4358c
run() |> display

# ╔═╡ 79bd0750-328b-4189-ad77-08cf4fe8fdae
to

# ╔═╡ Cell order:
# ╠═4e32423b-8a3a-45d6-b00c-276c7936dc5a
# ╟─f7847c56-f124-4716-b38c-4582752807d6
# ╠═e390926a-3b59-4019-90b3-578c493d7867
# ╠═685184f1-3d61-40d9-a30f-82a17d1e48c5
# ╠═5cd351d0-5635-486d-b890-61bfc7af7c46
# ╠═22c1871f-0a7f-425e-9194-ac6a26de6f4c
# ╠═2a3366e0-cf21-4ea6-a626-4789dd1ab002
# ╠═534e7802-63f3-43d5-99d6-2f25cd33b3ba
# ╠═63931b1e-fba0-49e8-a19f-9102ac5f144b
# ╠═829cb50a-90c0-4017-9533-2c28e11c7213
# ╠═cf0f8cfd-69d6-4b68-bdfd-8a8ca4dadc61
# ╠═2d55378b-b7cb-49b8-8222-422aa0d4358c
# ╠═79bd0750-328b-4189-ad77-08cf4fe8fdae
