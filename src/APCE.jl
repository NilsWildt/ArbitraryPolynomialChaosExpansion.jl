module APCE
using PrecompileTools: @setup_workload, @compile_workload    # this is a small dependency

using DrWatson
configdir(args...) = projectdir("configs", args...)
outputdir(args...) = projectdir("output", args...)

export run, normalization_functions, aPCE_MultivariatePolynomialDegrees, ∂∂I_aPCE_PsiPolynomialMatrix, RowVecs, ColVecs, GaussianCollocation, train!, predict, UQ, partitionTrainTest, aPCE, aPCE_OrthonormalBasis,create_Polynomial_Degrees,create_basis,evaluate_Ψ

include("APCEfunctions.jl")
include("APCEderivatives.jl")
include("APCEhighlevel.jl")
include("utils.jl")



@setup_workload begin
    FT = Float64
    TrainingInput = rand(25,2) |> Array{FT}
    TrainingOutput = rand(25,2) |> Array{FT} |> Array{FT}
    @compile_workload begin
        degree = 3
        apc_instance = aPCE(TrainingInput, degree; outdim = size(TrainingOutput, 2), OrthonormalRepresentation = true, qnorm = 0.7,normalize_data=true)
        train!(apc_instance, TrainingInput, TrainingOutput; bayesian_inversion = true, reg_mode=3)
        predict(apc_instance, TrainingInput)
        UQ(apc_instance)
    end
end


end
