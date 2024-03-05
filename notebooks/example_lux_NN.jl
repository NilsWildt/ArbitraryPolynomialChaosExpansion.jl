### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ e53c1b98-a12e-4d59-9326-76481c47a5e4
begin
	using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
	    OptimizationOptimisers, Random, Plots
	using Lux, LuxAMDGPU, LuxCUDA, Optimisers, Random, Statistics, Zygote
	using CairoMakie, MakiePublication
end

# ╔═╡ aab95b8a-d8cd-41e4-8a9a-f9e8fefa90b0
md""" 
### First example: Regression
"""

# ╔═╡ 5a6e9b3c-f2fa-4361-9b2a-f6d3e5464739


# ╔═╡ 0cdaf092-9926-44d6-8d15-d12b0cc39ea1
md""" 
### Second example: Neural ODE
"""

# ╔═╡ f7b8c800-dafc-11ee-2645-1d135484a413
# begin
	
	
# 	rng = Random.default_rng()
# 	u0 = Float32[2.0; 0.0]
# 	datasize = 30
# 	tspan = (0.0f0, 1.5f0)
# 	tsteps = range(tspan[1], tspan[2]; length = datasize)
	
# 	function trueODEfunc(du, u, p, t)
# 	    true_A = [-0.1 2.0; -2.0 -0.1]
# 	    du .= ((u .^ 3)'true_A)'
# 	end
	
# 	prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# 	ode_data = Array(solve(prob_trueode, Tsit5(); saveat = tsteps))
	
# 	dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
# 	p, st = Lux.setup(rng, dudt2)
# 	prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)
	
# 	function predict_neuralode(p)
# 	    Array(prob_neuralode(u0, p, st)[1])
# 	end
	
# 	function loss_neuralode(p)
# 	    pred = predict_neuralode(p)
# 	    loss = sum(abs2, ode_data .- pred)
# 	    return loss, pred
# 	end
	
# 	# Do not plot by default for the documentation
# 	# Users should change doplot=true to see the plots callbacks
# 	callback = function (p, l, pred; doplot = false)
# 	    println(l)
# 	    # plot current prediction against data
# 	    if doplot
# 	        plt = scatter(tsteps, ode_data[1, :]; label = "data")
# 	        scatter!(plt, tsteps, pred[1, :]; label = "prediction")
# 	        display(plot(plt))
# 	    end
# 	    return false
# 	end
	
# 	pinit = ComponentArray(p)
# 	callback(pinit, loss_neuralode(pinit)...; doplot = true)
	
# 	# use Optimization.jl to solve the problem
# 	adtype = Optimization.AutoZygote()
	
# 	optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
# 	optprob = Optimization.OptimizationProblem(optf, pinit)
	
# 	result_neuralode = Optimization.solve(optprob, Adam(0.05); callback = callback,
# 	    maxiters = 300)
	
# 	optprob2 = remake(optprob; u0 = result_neuralode.u)
	
# 	result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
# 	    callback, allow_f_increases = false)
	
# 	callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)
# end

# ╔═╡ Cell order:
# ╠═e53c1b98-a12e-4d59-9326-76481c47a5e4
# ╠═aab95b8a-d8cd-41e4-8a9a-f9e8fefa90b0
# ╠═5a6e9b3c-f2fa-4361-9b2a-f6d3e5464739
# ╠═0cdaf092-9926-44d6-8d15-d12b0cc39ea1
# ╠═f7b8c800-dafc-11ee-2645-1d135484a413
