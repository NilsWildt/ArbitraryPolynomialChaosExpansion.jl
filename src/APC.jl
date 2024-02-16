module APC
using DrWatson

configdir(args...) = projectdir("configs", args...)
outputdir(args...) = projectdir("output", args...)

export run
include(srcdir("utils.jl"))
include(srcdir("APC_types.jl"))
include(srcdir("APC_utils.jl"))
include(srcdir("experiment.jl"))
include(srcdir("APC_collocation.jl"))
include(srcdir("APC_ONB.jl"))
include(srcdir("APC_ONB_MV.jl"))

using Plots
plotly()
function run(degree)
	err = []
	ts = []
	ps = []
	# degress = 1:1:5
	N = 3000
	d = 2

	x = get_input(N, d, 1)
	# @debug "" x
	# m(x) = PhysicalModel1D(1,x)
	last_pred = undef


    true_output = [PhysicalModel1D(1, ix) for ix in x]


    if d == 2
        true_output = [PhysicalModelND(1, ix) for ix in x]
    end

	

	# degree = 2
	# t = @elapsed begin
	apc_instance = aPC(x, degree)
	# @info "" apc_instance 
	TrainingInput = GaussianCollocation(apc_instance)
	TrainingOutput = []
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
	TrainingOutput = reduce(hcat, TrainingOutput)' |> RowVecs

	# @info "" apc_instance	
	train!(apc_instance, TrainingInput, TrainingOutput)

	# # @debug "" UQ(apc_instance)
	# # xcp = [0.422117914428017	0.610608061392019	0.780825721677619]]


	pred = predict(apc_instance, x)
    @show mean(true_output)
    @show var(true_output)
    @show UQ(apc_instance)
	# display(pred)
	# last_pred = yy-> predict(apc_instance,[yy]) 
	# Plots.plot(x,TrainingOutput)
   
   
    # x = reduce(vcat,x)
    # p1 = Plots.scatter(x, true_output, ms = 1)
	# p2 = Plots.scatter(x, pred, ms = 1)
	# p3 = Plots.scatter(x, pred .- true_output, ms = 1)

	# p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 1)
	# p2 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 1)
	# p3 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred .- true_output, ms = 1)
 

    # p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 1, mc = :blue,size = (800, 800))
    # Plots.scatter!(p1,reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 1, mc = :red)
    # # p3 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred .- true_output, ms = 1, mc = :green)

    # display(p1)
    
    gratio = (1.0+sqrt(5.0))/2.0
    p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 2, mc = :blue, marker = :circle, label = "True Output", legend = true, size = (600*gratio,600))
    Plots.scatter!(p1,reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 2, mc = :red, marker = :square, label = "Prediction", legend = true)
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
end


end
