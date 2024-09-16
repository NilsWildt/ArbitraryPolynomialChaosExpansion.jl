using Revise
module Main
using DrWatson
using PropDicts
# using PProf
@quickactivate "APCE.jl"
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
using Suppressor

loggingdir(args...) = projectdir("output", "logs", args...)
mkpath(loggingdir())
# io = open(loggingdir("debug.txt"), "w")
# file_logger = ConsoleLogger(io, Logging.Debug)
debug_logging = ConsoleLogger(stderr, Logging.Debug)
info_logging = ConsoleLogger(stderr, Logging.Info)
# Here you may include files from the source directory
global_logger(debug_logging)
include(srcdir("APCE.jl"))
using .APC

using TimerOutputs
const to = TimerOutput()
mytimes = []
 begin # @suppress
	for d in 1:20
		# APC.run1(d, to)
		# display(degree)
		# degree = 25
		t = @elapsed begin
			APC.run3(d, to)
		end
		push!(mytimes, [d, t])
	end
end
display(to)
# APC.run(6)
# ProfileView.@profview APC.run(20,to)
mytimes = reduce(hcat, mytimes)
fig = Figure()
ax = Axis(fig[1, 1], yscale = log10, xlabel = "degree", ylabel = "time (s)")
# Plot the data
scatter!(ax, mytimes[1, :], mytimes[2, :], label = "Julia")
times_mat = matread(Base.Filesystem.normpath("C:/Users/wildt/Downloads/aPC Matlab Toolbox/aPC Matlab Toolbox (1)/aPC Matlab Toolbox/times.mat"))
scatter!(ax, times_mat["ds"] |> vec, times_mat["ts"] |> vec, color = :red, label = "Matlab", marker = :x)
legend = Legend(fig, ax, "Legend", valign = :top)
fig[1, 2] = legend
display(fig)
# save("comparison.png", fig)
end
