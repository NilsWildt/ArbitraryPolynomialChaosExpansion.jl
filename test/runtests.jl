using DrWatson, Test
@quickactivate "APCE"
using APCE
using PerfChecker
using Aqua



@testset verbose = true showtiming = true "All tests" begin
	@testset verbose = true "Aqua.test_all" begin
        Aqua.test_all(APCE) |> display
	end

end
