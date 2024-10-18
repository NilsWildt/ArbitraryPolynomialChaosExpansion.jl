using Revise
module Main
    using DrWatson
    using PropDicts
    # using PProf
    @quickactivate "APCE"
    using Logging
    using TerminalLoggers: TerminalLogger
    using ProgressLogging
    using Logging
    using LoggingExtras
    using PProf
    using BenchmarkTools
    # using ProfileView
    using Makie
    using CairoMakie
    using MAT
    using Plots
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
    @timev p = APC.run1(3, to)
    display(to)
    display(p)

    # @timev p = APC.run3(10,to)

    # display(to)

    # current()
end
