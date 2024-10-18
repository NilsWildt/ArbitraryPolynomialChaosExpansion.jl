using Revise
using DrWatson
@quickactivate "APCE"
module Runner
    using DrWatson
    using PrettyTables
    using DifferentiationInterfaceTest
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
    using Optim
    using ParameterSchedulers
    using Pkg
    using PProf
    using PrettyTables
    using ProgressMeter
    using PolyesterForwardDiff: PolyesterForwardDiff
    using FastDifferentiation: FastDifferentiation
    using Diffractor: Diffractor
    using SparseArrays
    # using SparseDiffTools
    using UnicodePlots
    using Zygote: Zygote
    using TimerOutputs
    using Tapir: Tapir
    using DifferentiationInterface
    using Enzyme: Enzyme
    using FastDifferentiation: FastDifferentiation
    using ForwardDiff: ForwardDiff
    using Zygote: Zygote
    using Tapir: Tapir
    using FiniteDiff: FiniteDiff
    using ReverseDiff: ReverseDiff
    using FastDifferentiation: FastDifferentiation
    using DataFrames: DataFrames
    using Markdown: Markdown
    using PrettyTables: PrettyTables
    using Printf: Printf
    # import Lux.Experimental.
    # using Phi
    # using Tracker
    # using Enzyme
    Enzyme.API.runtimeActivity!(true)
    # using Test
    Zygote.refresh()

    loggingdir(args...) = projectdir("output", "logs", args...)
    mkpath(loggingdir())
    # io = open(loggingdir("debug.txt"), "w")
    # file_logger = ConsoleLogger(io, Logging.Debug)
    debug_logging = ConsoleLogger(stderr, Logging.Debug)
    info_logging = ConsoleLogger(stderr, Logging.Info)
    # Here you may include files from the source directory
    global_logger(debug_logging)
    include(srcdir("APCE.jl"))
    using .APCE
    # using Preferences
    # set_preferences!(APCE, "precompile_workload" => false; force=true)

    using TimerOutputs
    const to = TimerOutput()

    function aPCE_init_weights_0_1(P, in_dims, out_dims; is_sparse = false)
        if is_sparse
            weights = spzeros(P, out_dims)
            weights[1:(in_dims + 1), :] .= 1.0
            return weights
        else
            weights = zeros(P, out_dims) # spzeros
            weights[1:(in_dims + 1), :] .= 1.0
            return weights
        end
    end


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
            file = matread(datadir("gw_training_data.mat"))
            print(keys(file))
            # Extracting the variables
            # Input_distributions = file["Xtr"] |> Array{FT}
            # @info size(file["Xtr"])
            indall = 1:10
            rng = Xoshiro(42)
            # @info "" file["TrainingOutput"]
            itrain, itest = partitionTrainTest(indall; at = 0.75, rng)
            TrainingInput = file["TrainingInput"][indall, 1:5] |> Array{FT}
            TrainingOutput = file["TrainingOutput"][indall, 1:5] |> Array{FT}
            # @info "" size(TrainingInput)

            ValidationInput = file["ValidationInput"][indall, 1:5] |> Array{FT}
            ValidationOutput = file["ValidationOutput"][indall, 1:5] |> Array{FT}
        end

        # Display the table


        ℓ2(x) = mean(abs2, x)
        ℓ1(x) = mean(abs, x)

        d_expansion = 3
        in_dims = size(TrainingInput, 2)
        out_dims = size(TrainingOutput, 2)
        dx, dy = max(d_expansion, in_dims), min(d_expansion, in_dims)
        P = UInt128(prod(UInt128(dx + 1):UInt128(d_expansion + in_dims)) ÷ factorial(UInt128(dy))) |> Int
        MultivariatePolynomialDegrees = create_Polynomial_Degrees(in_dims, d_expansion; qnorm = 1.0)
        P = min(size(MultivariatePolynomialDegrees, 1), P)
        OrthonormalBasis = create_basis(TrainingInput, d_expansion)
        #

        function compute_loss(expansion_coefficients, x, y, xvalid, yvalid, d_expansion = d_expansion)
            out = evaluate_Ψ(x, expansion_coefficients, MultivariatePolynomialDegrees, OrthonormalBasis, d_expansion)
            return sqrt(mean(((out .- ValidationOutput) .^ 2)))
        end

        # cps = ComponentVector(ps)
        # ax = getaxes(cps)
        # flat = getdata(cps)
        # scatterplot(1:length(flat), flat) |> display

        flat = aPCE_init_weights_0_1(P, in_dims, out_dims)
        function obj(θ::AbstractArray{T}) where {T <: Real}
            loss = compute_loss(θ, TrainingInput, TrainingOutput, ValidationInput, ValidationOutput, d_expansion)
            return [loss]
        end

        ∇f(θ::AbstractArray) = ForwardDiff.jacobian(θ -> obj(θ), θ)

        backends = [AutoFiniteDiff(), AutoReverseDiff(), AutoReverseDiff(; compile = true), AutoZygote(), AutoForwardDiff(), AutoTracker()] #AutoTapir() ,  AutoEnzyme(; mode = Enzyme.Reverse), AutoEnzyme(; mode = Enzyme.Forward) AutoReverseDiff(), AutoZygote(), AutoForwardDiff(),AutoTapir(), AutoEnzyme(; mode=Enzyme.Reverse), AutoEnzyme(; mode=Enzyme.Forward)
        scenarios = [
            JacobianScenario(obj; x = Float64.(flat), ref = ∇f),
            # JacobianScenario(obj; x = Float32.(flat), ref = ∇f),
            # JacobianScenario(obj; x = Float16.(flat), ref = ∇f)
        ]
        @info ""
        # JacobianScenario(obj; x=Float32.(flat), ref=∇f)

        test_differentiation(
            backends,  # the backends you want to compare
            scenarios,  # the scenarios you defined,.
            correctness = true,  # compares values against the reference
            type_stability = false,  # checks type stability with JET.jl
            detailed = true,  # prints a detailed test set
        )

        benchmark_result = benchmark_differentiation(backends, scenarios)
        df = DataFrames.DataFrame(benchmark_result)
        function formatter(v, i, j)
            if j in (15, 16)  # time, bytes
                return Printf.@sprintf("%.1e", v)
            elseif j == 17  # allocs
                return Printf.@sprintf("%.1f", v)
            else
                return v
            end
        end
        sort!(df, [:time])
        table = PrettyTables.pretty_table(
            String,
            df;
            backend = Val(:markdown),
            header = names(df),
            formatters = formatter,
        )
        return Markdown.parse(table) |> display
    end


    #####################

    run()
end
