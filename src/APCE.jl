module APCE
using DrWatson
configdir(args...) = projectdir("configs", args...)
outputdir(args...) = projectdir("output", args...)

export run, normalization_functions, aPCE_MultivariatePolynomialDegrees, ∂∂I_aPCE_PsiPolynomialMatrix, RowVecs, ColVecs, GaussianCollocation, train!, predict, UQ, partitionTrainTest, aPCE, aPCE_OrthonormalBasis,create_Polynomial_Degrees,create_basis,evaluate_Ψ

include("APCEfunctions.jl")
include("APCEderivatives.jl")
include("APCEhighlevel.jl")
include("utils.jl")
end
