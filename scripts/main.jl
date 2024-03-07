using Revise
module Main
using DrWatson
using PropDicts
# using PProf
@quickactivate "aPC.jl"
using Logging
using TerminalLoggers: TerminalLogger
using ProgressLogging
using Logging
using LoggingExtras
using PProf
using BenchmarkTools
using ProfileView

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

for _ in 1:5
# display(degree)
	degree = 25
	@timeit to "APC" begin
		p = APC.run(degree, to)
	end
	end
display(to)
# APC.run(6)
ProfileView.@profview APC.run(20,to)
end