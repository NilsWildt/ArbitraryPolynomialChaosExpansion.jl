@show "DaPC NN"
using Pkg
#Pkg.add("MAT")
#Pkg.add("MakiePublication")
using MAT
using Lux, Optimisers, Random, Statistics, Zygote
using CairoMakie, MakiePublication
include("PolyLayer.jl")
#  Loading Input distributions, Training Data and Validation Data
file = matread(raw"Z:\users\ac125867\03_projects\32_aPC_julia\aPC.jl\notebooks\Data.mat")

# Extracting the variables
Input_distributions = file["Input_distributions"]
TrainingInput = file["TrainingInput"]
TrainingOutput = file["TrainingOutput"]
ValidationInput = file["ValidationInput"]
ValidationOutput = file["ValidationOutput"]

# Initialzation of Deep aPC Neural Netwok
model = Chain(Dense(3 => 10, relu), Dense(10 => 1))
rng = MersenneTwister()
Random.seed!(rng, 12345)
opt = Adam(0.3f0)
NN = Lux.Training.TrainState(rng, model, opt)

# NN Run
function model_run(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    LossFunction = mean(abs2, y_pred .- data[2]) #Loss function
    return LossFunction, st, ()
end

# Library for the Derivatives
vjp_rule = Lux.Training.AutoZygote()

function main(NN::Lux.Experimental.TrainState, vjp, data, epochs)
    for epoch in 1:epochs
        grads, loss, stats, NN = Lux.Training.compute_gradients(vjp, model_run, data, NN) # Derivative
        println("Epoch: $(epoch) || Loss: $(loss)")
        NN = Lux.Training.apply_gradients(NN, grads) #Optimization
    end
    return NN
end

dev_cpu = cpu_device()
#dev_gpu = gpu_device()

NN = main(NN, vjp_rule, (TrainingInput', TrainingOutput'), 250)
y_pred = dev_cpu(Lux.apply(NN.model, TrainingInput', NN.parameters, NN.states)[1])
