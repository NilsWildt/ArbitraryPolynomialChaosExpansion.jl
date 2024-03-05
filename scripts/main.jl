using Revise
module Main
using DrWatson
using PropDicts
# using PProf
using DrWatson
@quickactivate "aPC.jl"
using Logging
using TerminalLoggers: TerminalLogger
using ProgressLogging
using Logging
using LoggingExtras
using PProf


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

# for degree in 2:2
	# display(degree)
# 	for l in 1:100
# 	degree = 3
# 	@timeit to "APC" begin
# 		p = APC.run(degree)
# 	end
# end
# end
# display(to)

APC.run(6)

end