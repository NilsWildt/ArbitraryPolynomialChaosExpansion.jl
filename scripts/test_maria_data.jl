using Revise
using DrWatson
# using PProf
@quickactivate "aPC.jl"
module Runner
using DrWatson
# import Pkg 
using PrettyTables
# Pkg.instantiate()
using PropDicts
# using PProf
using Logging
using TerminalLoggers: TerminalLogger
using ProgressLogging
using Logging
using LoggingExtras
# using PProf
using BenchmarkTools
# using ProfileView
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
global_logger(debug_logging)
include(srcdir("APC.jl"))
using .APC
using TimerOutputs
const to = TimerOutput()


function run()
	err = []
	ts = []
	ps = []
	# degress = 1:1:5
	

	FT = Float64
	@timeit to "lod_data" begin
		#  Loading Input distributions, Training Data and Validation Data 
		file = matread(datadir("training_data.mat"))
		print(keys(file))
		# Extracting the variables
		# Input_distributions = file["Xtr"] |> Array{FT}
		indall = 1:1000
		rng = Xoshiro(42)

		itrain,itest = partitionTrainTest(indall;at= 0.75, rng )
		TrainingInput = file["Xtr"][itrain, :] |> Array{FT}
		TrainingOutput = file["Ytr"][itrain, 60:62] |> Array{FT}
		@info "" size(TrainingInput)

		ValidationInput = file["Xtr"][itest, :] |> Array{FT}
		ValidationOutput = file["Ytr"][itest, 60:62] |> Array{FT}
	end
	degree = 7
	@info "" size(TrainingOutput, 2)
	apc_instance = @timeit to "aPC_instance" aPC(TrainingInput, degree; outdim = size(TrainingOutput, 2), OrthonormalRepresentation = true, qnorm = 0.3,normalize_data=true)
	@assert apc_instance.NumberOfTerms<2000 "Too many coefficients."

	# TrainingInput = GaussianCollocation2(apc_instance)
	# @debug "" TrainingInput
	# display(	@report_call GaussianCollocation(apc_instance; strategy = :PCM) )
	# @descend GaussianCollocation(apc_instance; strategy = :PCM)
	# TrainingInput = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy = :PCM)
	# TrainingInput = @timeit to "KMeansCollocation" KMeansCollocation(apc_instance)
	@info "" size(TrainingInput)
	@timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true,reg_mode=3)


	PredictionOutput = @timeit to "prediction" predict(apc_instance, TrainingInput)
	ValidationPredictionOutput = @timeit to "prediction" predict(apc_instance, ValidationInput)


	uq = UQ(apc_instance)
	data = ["Mean" uq.OutputMean[1] mean(ValidationOutput); "Var" uq.OutputVar[1] var(ValidationOutput)]

	headers = ["Type", "aPCE", "Data"]

	# Display the table
	pretty_table(data; header = headers)


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
run()
end