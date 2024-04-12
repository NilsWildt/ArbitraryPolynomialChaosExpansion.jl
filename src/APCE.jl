module APCE
using DrWatson
configdir(args...) = projectdir("configs", args...)
outputdir(args...) = projectdir("output", args...)

export run, normalization_functions, aPCE_MultivariatePolynomialDegrees, ∂∂I_aPCE_PsiPolynomialMatrix, RowVecs, ColVecs, GaussianCollocation, train!, predict, UQ, partitionTrainTest, aPCE, aPCE_OrthonormalBasis

include(srcdir("APCEfunctions.jl"))
include(srcdir("APCEderivatives.jl"))
include(srcdir("APCEhighlevel.jl"))
include(srcdir("utils.jl"))
end
