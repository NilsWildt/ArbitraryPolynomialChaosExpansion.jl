# MooncakeExt.jl - Extension for Mooncake AD support
# This extension is loaded when Mooncake is available

module MooncakeExt

using ArbitraryPolynomialChaosExpansion
using LinearAlgebra: pinv
using Mooncake: @from_rrule, DefaultCtx

# Register ChainRules with Mooncake for create_basis
@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.create_basis),
    AbstractArray, Integer,
}

@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.create_basis),
    AbstractArray, Integer, Val,
}

@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.create_basis),
    AbstractArray, Integer, Bool,
}

@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.aPCE_OrthonormalBasis),
    AbstractArray, Integer, Val,
}

@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.create_centered_basis),
    AbstractArray, Integer,
}

@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.create_recurrence_basis),
    AbstractArray, Integer,
}

# Register multires basis construction (AD-opaque: NoTangent)
@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.create_multiwavelet_basis),
    AbstractArray, Integer,
}

@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.create_multiwavelet_basis),
    AbstractArray, Vector{Int},
}

# Register multires Psi assembly (custom analytic rrule)
@from_rrule DefaultCtx Tuple{
    typeof(ArbitraryPolynomialChaosExpansion.aPCE_PsiPolynomialMatrix_zygote),
    AbstractArray, AbstractMatrix{Int}, ArbitraryPolynomialChaosExpansion.MultiWaveletBasis,
}

@from_rrule DefaultCtx Tuple{typeof(ArbitraryPolynomialChaosExpansion.solve_linear_robust), Any, Any} true
@from_rrule DefaultCtx Tuple{typeof(pinv), AbstractMatrix}

end # module
