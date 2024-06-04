using DrWatson, Test
# @quickactivate "APCE"
using APCE
using PerfChecker
using Aqua
using JET
using Revise
using DrWatson
# using PrettyTables
using BenchmarkTools
using MethodAnalysis
using JET
# loggingdir(args...) = projectdir("output", "logs", args...)
# mkpath(loggingdir())
# # io = open(loggingdir("debug.txt"), "w")
# # file_logger = ConsoleLogger(io, Logging.Debug)
# debug_logging = ConsoleLogger(stderr, Logging.Debug)
# info_logging = ConsoleLogger(stderr, Logging.Info)
# # Here you may include files from the source directory
# global_logger(info_logging)
include(srcdir("../../src/APCE.jl"))
# using .APCE
# import .APCE: train!
# import .APCE: predict
# import .APCE: UQ

# using Preferences
# set_preferences!(APCE, "precompile_workload" => false; force=true)
using TimerOutputs

report_package("APCE")

# JET.report_file(normpath(raw"\\iws-ls3-cifs.tik.uni-stuttgart.de\shared\users\ac125867\03_projects\32_aPC_julia\APCE.jl\examples\test_maria_data.jl"))

begin
	mis = methodinstances(APCE)    # get all the compiled methodinstances for functions owned by the package
	# Now let's filter out the ones that pass without issue
	badmis = filter(mis) do mi
		!isempty(JET.get_reports(report_call(mi)))
		# JET.get_reports(report_call(mi))
	end
	@info badmis
end


@testset verbose = true showtiming = true "All tests" begin
	@testset verbose = true "Aqua.test_all" begin
		Aqua.test_all(APCE) |> display
	end

end
