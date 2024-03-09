module APC
using DrWatson


configdir(args...) = projectdir("configs", args...)
outputdir(args...) = projectdir("output", args...)

export run
include(srcdir("utils.jl"))
include(srcdir("experiment.jl"))
include(srcdir("APC_functions.jl"))
include(srcdir("analyticalmodel.jl"))


using Plots
plotly()
using TimerOutputs

# using TimerOutputs
# const to = TimerOutput()

# using Cthulhu
# using JET 

function meshgrid(x, y)
	X = [x for _ in y, x in x]
	Y = [y for y in y, _ in x]
	X, Y
end


function run(degree, to, data)
	TrainingInput = data["TrainingInput"][:,1:13]
	TrainingOutput = data["TrainingOutput"][:,1:4]
	true_output = data["Input_distributios"][:,1:4]

	TrainingInput = RowVecs(TrainingInput)
	TrainingOutput = RowVecs(TrainingOutput)
	apc_instance = @timeit to "aPC_instance" aPC(TrainingInput, degree; OrthonormalRepresentation = true, qnorm = 1.0)
	# TrainingInput = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy = :PCM)
	# TrainingOutput = reduce(hcat, TrainingOutput)' |> RowVecs
	@timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput;bayesian_inversion=true)

	pred = @timeit to "prediction" predict(apc_instance, TrainingInput)
# @info size(pred) size( TrainingInput)
	for k in 1:apc_instance.output_dimensions
		uq = UQ(apc_instance;axis=k)
		data_table = ["Mean" uq.OutputMean[1] mean(true_output[:,k]); "Var" uq.OutputVar[1] var(true_output[:,k])]
	
		headers = ["Type", "aPCE", "Data"]
	
		# Display the table
		pretty_table(data_table; header = headers)
	end

	gratio = (1.0 + sqrt(5.0)) / 2.0
	ps = []
	for k in axes(pred,2)
		
	p1 = Plots.scatter(reduce(hcat,TrainingInput)[1,:], reduce(hcat,TrainingInput)[2,:],reduce(hcat,TrainingOutput)[k,:], ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.3)
	Plots.scatter!(p1, reduce(hcat,TrainingInput)[1,:], reduce(hcat,TrainingInput)[2,:],pred[:,k], ms = 2, mc = :red, marker = :square, label = "Prediction", legend = true, alpha = 1.0)

	Plots.scatter!(p1, reduce(hcat,TrainingInput)[1,:],reduce(hcat,TrainingInput)[2,:],reduce(hcat,TrainingOutput)[k,:], ms = 2, mc = :green, marker = :square, label = "Collocation Output", legend = true)

		push!(ps,p1)
	end


	return Plots.plot(ps...)
end

function run3(degree, to)

	# degress = 1:1:5
	N = 1000
	d = 2


	x = @timeit to "get_input" get_input(N, d, 1)
	# @debug "" x
	# m(x) = PhysicalModel1D(1,x)
	# last_pred = undef

	true_output = [PhysicalModel1D(1, ix) for ix in x]
	if d == 2
		true_output = [PhysicalModelND(1, ix) for ix in x]
	end

	true_output = reduce(vcat,true_output)
	# degree = 2
	# t = @elapsed begin
	apc_instance = @timeit to "aPC_instance" aPC(x, degree; OrthonormalRepresentation = true, qnorm = 1.0)
	# @info "" apc_instance 

	# TrainingInput = GaussianCollocation2(apc_instance)
	# @debug "" TrainingInput
	# display(	@report_call GaussianCollocation(apc_instance; strategy = :PCM) )
	# @descend GaussianCollocation(apc_instance; strategy = :PCM)
	TrainingInput = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy = :PCM)
	# TrainingInput = @timeit to "KMeansCollocation" KMeansCollocation(apc_instance)

	# @debug "" TrainingInput
	# (xg, yg) = meshgrid(LinRange(0, 1, 50), LinRange(0, 1, 50))

	# TrainingInput = []
	# for i in axes(xg,1)
	# 	push!(TrainingInput,[xg[i],yg[i]])
	# end
	# display(TrainingInput)
	# TrainingInput = RowVecs(reduce(hcat,TrainingInput))
	# display(TrainingInput)


	TrainingOutput = Array{Float64}[]
	for i ∈ 1:size(TrainingInput, 1)
		if d == 2
			# @show TrainingInput[i]
			push!(TrainingOutput, PhysicalModelND(1, TrainingInput[i]))
		elseif d == 1
			# @show typeof(TrainingInput[i][1])
			push!(TrainingOutput, PhysicalModel1D(1, vec(TrainingInput[i])[1]))
		end
	end
	# @show TrainingOutput
	# @info "" size(TrainingOutput) TrainingOutput

	TrainingOutput = reduce(hcat,TrainingOutput)' |> RowVecs

	# @info "" size(TrainingOutput) TrainingOutput

	# @descend train!(apc_instance, TrainingInput, TrainingOutput)
	# @info "" apc_instance	
	@timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput;bayesian_inversion=false)

	# # @debug "" UQ(apc_instance)
	# # xcp = [0.422117914428017	0.610608061392019	0.780825721677619]]

	# @descend predict(apc_instance, x)
	pred = @timeit to "prediction" predict(apc_instance, x)
	# @show mean(true_output)
	# @show var(true_output)
	# @show UQ(apc_instance)
	# display(pred)
	# last_pred = yy-> predict(apc_instance,[yy]) 
	# Plots.plot(x,TrainingOutput)


	# x = reduce(vcat, x)
	# gratio = (1.0 + sqrt(5.0)) / 2.0
	# p1 = Plots.scatter(x, true_output, ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.8)
	# Plots.scatter!(p1, x, pred, ms = 2, mc = :red, marker = :square, label = "Prediction", legend = true, alpha = 0.8)
	# Plots.scatter!(p1, TrainingInput, TrainingOutput, ms = 2, mc = :green, marker = :square, label = "Collocation", legend = true, alpha = 0.8)

	# p2 = Plots.scatter(x, pred, ms = 1)
	# p3 = Plots.scatter(x, pred .- true_output, ms = 1)

	# p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 1)
	# p2 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 1)
	# p3 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred .- true_output, ms = 1)


	# p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 1, mc = :blue,size = (800, 800))
	# Plots.scatter!(p1,reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 1, mc = :red)
	# # p3 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred .- true_output, ms = 1, mc = :green)

	# display(p1)
	# @info mean(true_output)
	# @info var(true_output)
	uq = UQ(apc_instance)
	data = ["Mean" uq.OutputMean[1] mean(true_output); "Var" uq.OutputVar[1] var(true_output)]

	headers = ["Type", "aPCE", "Data"]

	# Display the table
	pretty_table(data; header = headers)

	gratio = (1.0 + sqrt(5.0)) / 2.0
	# @info "" reduce(hcat, TrainingInput)[1, :] reduce(hcat, TrainingInput)[2, :] reduce(vcat, TrainingOutput) reduce(hcat, x)[1, :] reduce(hcat, x)[2, :] pred 
	# p1 = Plots.scatter(reduce(hcat, x)[1, :],  reduce(hcat, x)[2, :], vec(true_output), ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.3)
	# Plots.scatter!(p1, reduce(hcat, TrainingInput)[1, :] ,reduce(hcat, TrainingInput)[2, :], reduce(vcat, TrainingOutput), ms = 2, mc = :red, marker = :square, label = "Trainingpoints", legend = true, alpha = 0.3)

	# Plots.scatter!(p1, reduce(hcat, TrainingInput)[1, :], reduce(hcat, TrainingInput)[2, :], reduce(vcat, pred), ms = 2, mc = :green, marker = :square, label = "Predicted On Training Points", legend = true)


	# 	p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.3)
	# Plots.scatter!(p1, reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], vec(pred), ms = 2, mc = :red, marker = :square, label = "Prediction", legend = true, alpha = 0.3)

	# Plots.scatter!(p1, reduce(hcat, TrainingInput)[1, :], reduce(hcat, TrainingInput)[2, :], reduce(vcat, TrainingOutput), ms = 2, mc = :green, marker = :square, label = "Collocation Output", legend = true)


	# Plots.scatter!(p2,reduce(hcat,TrainingInput)[2,:],)
	# expect = RowVecs(PhysicalModel1D.(1,TrainingInput))
	# display(expect)
	# Plots.plot(p1, p2, p3,size = (800, 800))

	# # @debug "" size(expect) size(pred)
	# push!(err,maximum(abs.(pred.-expect).^2))
	# 		@debug expect.-pred
	# push!(ts,t)

	# 		xtest = LinRange(-1,1,50)
	# p3 = Plots.scatter(xtest, m.(xtest')',alpha=0.5)
	# preds = [last_pred(x)|>first for x in xtest]
	# @show preds
	# Plots.scatter!(p3,xtest, preds,alpha=0.5)

	# 	push!(ps,p3)


	# p1 = Plots.scatter(degress, log.(ts))	
	# p2 = Plots.scatter(degress, log.(err))


	# pn = Plots.plot(p1,p2)
	# pn2 = Plots.plot(ps...)
	# Plots.plot(pn,pn2, layout =@layout [a ; b])
	# Plots.scatter(log.((abs.(pred[1,:].-expect[1,:]))))
	# display(to)
	# return p1
	nothing
end

function run1(degree, to)

	err = []
	ts = []
	ps = []
	# degress = 1:1:5
	N = 3000
	d = 2


	x = @timeit to "get_input" get_input(N, d, 1)
	# @debug "" x
	# m(x) = PhysicalModel1D(1,x)
	# last_pred = undef

	true_output = [PhysicalModel1D(1, ix) for ix in x]
	if d == 2
		true_output = [PhysicalModelND(1, ix) for ix in x]
	end

	true_output = reduce(vcat,true_output)
	# degree = 2
	# t = @elapsed begin
	apc_instance = @timeit to "aPC_instance" aPC(x, degree;outdim=1, OrthonormalRepresentation = true, qnorm = 1.0)
	# @info "" apc_instance 

	# TrainingInput = GaussianCollocation2(apc_instance)
	# @debug "" TrainingInput
	# display(	@report_call GaussianCollocation(apc_instance; strategy = :PCM) )
	# @descend GaussianCollocation(apc_instance; strategy = :PCM)
	TrainingInput = @timeit to "GaussianCollocation" GaussianCollocation(apc_instance; strategy = :PCM)
	# TrainingInput = @timeit to "KMeansCollocation" KMeansCollocation(apc_instance)

	# @debug "" TrainingInput
	# (xg, yg) = meshgrid(LinRange(0, 1, 50), LinRange(0, 1, 50))

	# TrainingInput = []
	# for i in axes(xg,1)
	# 	push!(TrainingInput,[xg[i],yg[i]])
	# end
	# display(TrainingInput)
	# TrainingInput = RowVecs(reduce(hcat,TrainingInput))
	# display(TrainingInput)


	TrainingOutput = Array{Float64}[]
	for i ∈ 1:size(TrainingInput, 1)
		if d == 2
			# @show TrainingInput[i]
			push!(TrainingOutput, PhysicalModelND(1, TrainingInput[i]))
		elseif d == 1
			# @show typeof(TrainingInput[i][1])
			push!(TrainingOutput, PhysicalModel1D(1, vec(TrainingInput[i])[1]))
		end
	end
	# @show TrainingOutput
	# @info "" size(TrainingOutput) TrainingOutput

	TrainingOutput = reduce(hcat,TrainingOutput)' |> RowVecs

	# @info "" size(TrainingOutput) TrainingOutput

	# @descend train!(apc_instance, TrainingInput, TrainingOutput)
	# @info "" apc_instance	
	@timeit to "training" train!(apc_instance, TrainingInput, TrainingOutput;bayesian_inversion=true)

	# # @debug "" UQ(apc_instance)
	# # xcp = [0.422117914428017	0.610608061392019	0.780825721677619]]

	# @descend predict(apc_instance, x)
	pred = @timeit to "prediction" predict(apc_instance, x)
	# @show mean(true_output)
	# @show var(true_output)
	# @show UQ(apc_instance)
	# display(pred)
	# last_pred = yy-> predict(apc_instance,[yy]) 
	# Plots.plot(x,TrainingOutput)


	# x = reduce(vcat, x)
	# gratio = (1.0 + sqrt(5.0)) / 2.0
	# p1 = Plots.scatter(x, true_output, ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.8)
	# Plots.scatter!(p1, x, pred, ms = 2, mc = :red, marker = :square, label = "Prediction", legend = true, alpha = 0.8)
	# Plots.scatter!(p1, TrainingInput, TrainingOutput, ms = 2, mc = :green, marker = :square, label = "Collocation", legend = true, alpha = 0.8)

	# p2 = Plots.scatter(x, pred, ms = 1)
	# p3 = Plots.scatter(x, pred .- true_output, ms = 1)

	# p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 1)
	# p2 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 1)
	# p3 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred .- true_output, ms = 1)


	# p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 1, mc = :blue,size = (800, 800))
	# Plots.scatter!(p1,reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 1, mc = :red)
	# # p3 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred .- true_output, ms = 1, mc = :green)

	# display(p1)
	# @info mean(true_output)
	# @info var(true_output)
	uq = UQ(apc_instance)
	data = ["Mean" uq.OutputMean[1] mean(true_output); "Var" uq.OutputVar[1] var(true_output)]

	headers = ["Type", "aPCE", "Data"]

	# Display the table
	pretty_table(data; header = headers)

	gratio = (1.0 + sqrt(5.0)) / 2.0
	# @info "" reduce(hcat, TrainingInput)[1, :] reduce(hcat, TrainingInput)[2, :] reduce(vcat, TrainingOutput) reduce(hcat, x)[1, :] reduce(hcat, x)[2, :] pred 
	# p1 = Plots.scatter(reduce(hcat, x)[1, :],  reduce(hcat, x)[2, :], vec(true_output), ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.3)
	# Plots.scatter!(p1, reduce(hcat, TrainingInput)[1, :] ,reduce(hcat, TrainingInput)[2, :], reduce(vcat, TrainingOutput), ms = 2, mc = :red, marker = :square, label = "Trainingpoints", legend = true, alpha = 0.3)

	# Plots.scatter!(p1, reduce(hcat, TrainingInput)[1, :], reduce(hcat, TrainingInput)[2, :], reduce(vcat, pred), ms = 2, mc = :green, marker = :square, label = "Predicted On Training Points", legend = true)


		p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.3)
	Plots.scatter!(p1, reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], vec(pred), ms = 2, mc = :red, marker = :square, label = "Prediction", legend = true, alpha = 0.3)

	Plots.scatter!(p1, reduce(hcat, TrainingInput)[1, :], reduce(hcat, TrainingInput)[2, :], reduce(vcat, TrainingOutput), ms = 2, mc = :green, marker = :square, label = "Collocation Output", legend = true)


	# Plots.scatter!(p2,reduce(hcat,TrainingInput)[2,:],)
	# expect = RowVecs(PhysicalModel1D.(1,TrainingInput))
	# display(expect)
	# Plots.plot(p1, p2, p3,size = (800, 800))

	# # @debug "" size(expect) size(pred)
	# push!(err,maximum(abs.(pred.-expect).^2))
	# 		@debug expect.-pred
	# push!(ts,t)

	# 		xtest = LinRange(-1,1,50)
	# p3 = Plots.scatter(xtest, m.(xtest')',alpha=0.5)
	# preds = [last_pred(x)|>first for x in xtest]
	# @show preds
	# Plots.scatter!(p3,xtest, preds,alpha=0.5)

	# 	push!(ps,p3)


	# p1 = Plots.scatter(degress, log.(ts))	
	# p2 = Plots.scatter(degress, log.(err))


	# pn = Plots.plot(p1,p2)
	# pn2 = Plots.plot(ps...)
	# Plots.plot(pn,pn2, layout =@layout [a ; b])
	# Plots.scatter(log.((abs.(pred[1,:].-expect[1,:]))))
	# display(to)
	return p1
end



function run2(degree)
	err = []
	ts = []
	ps = []
	# degress = 1:1:5
	N = 500
	d = 2

	# locations
	(xg, yg) = meshgrid(LinRange(0, 1, 50), LinRange(0, 1, 50))
	# parameters

	true_output = [PhysicalModel1D(1, ix) for ix in x]

	if d == 2
		true_output = [PhysicalModel2DGaussian(1, ix) for ix in x]
	end

	# degree = 2
	# t = @elapsed begin
	apc_instance = aPC(x, degree)
	# @info "" apc_instance 

	# TrainingInput = GaussianCollocation2(apc_instance)
	# @debug "" TrainingInput
	TrainingInput = GaussianCollocation(apc_instance; strategy = :PCM)

	# 




	TrainingOutput = reduce(hcat, TrainingOutput)' |> RowVecs


	train!(apc_instance, TrainingInput, TrainingOutput)

	pred = predict(apc_instance, x)
	@info mean(true_output)
	@info var(true_output)
	@show UQ(apc_instance)

	# gratio = (1.0 + sqrt(5.0)) / 2.0


	gratio = (1.0 + sqrt(5.0)) / 2.0
	p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600 * gratio, 600), alpha = 0.3)
	Plots.scatter!(p1, reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 2, mc = :red, marker = :square, label = "Prediction", legend = true, alpha = 0.3)

	Plots.scatter!(p1, reduce(hcat, TrainingInput)[1, :], reduce(hcat, TrainingInput)[2, :], reduce(vcat, TrainingOutput), ms = 2, mc = :green, marker = :square, label = "Collocation Output", legend = true)
	p1
end


end
