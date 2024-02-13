# using DrWatson, Test
# @quickactivate "aPC.jl"

# include(srcdir("utils.jl"))
# include(srcdir("APC_types.jl"))
# include(srcdir("APC_utils.jl"))
# include(srcdir("experiment.jl"))
# include(srcdir("APC_collocation.jl"))
# include(srcdir("APC_ONB.jl"))
# include(srcdir("APC_ONB_MV.jl"))

# @testset verbose = true showtiming = true "All tests" begin
# 	@testset "aPC_OrthonormalBasis" begin
# 		@test aPC_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1) ≈ SMatrix{3, 3}(1, -0.5, -2.0, 0.0, 1.5, -1.5, 0.0, 0.0, 4.5)
# 	end

# 	@testset "aPC_MultivariatePolynomialDegrees" begin
# 		@test aPC_MultivariatePolynomialDegrees(2, 1) == SMatrix{3, 2}(0, 0, 1, 0, 1, 0)
# 		@test aPC_MultivariatePolynomialDegrees(2, 2) == SMatrix{6, 2}(0, 0, 1, 0, 1, 2, 0, 1, 0, 2, 1, 0)
# 	end

# end