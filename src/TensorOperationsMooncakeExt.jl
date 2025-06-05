module TensorOperationsMooncakeExt

using TensorOperations
using ChainRulesCore
using Mooncake
using Mooncake: @from_rrule, DefaultCtx, is_primitive

# Mark non-differentiable functions as primitives that return NoRData
for f in [
        :tensorstructure, :tensoradd_structure, :tensoradd_type,
        :tensoralloc_add, :tensorcontract_structure, :tensorcontract_type,
        :tensoralloc_contract, :ncontree, :nconoutput, :isnconstyle,
        :indexordertree,
    ]
    @eval begin
        Mooncake.is_primitive(::Type{<:DefaultCtx}, ::Type{<:Tuple{typeof(TensorOperations.$f), Vararg}}) = true
        Mooncake.rrule!!(f::Mooncake.CoDual{typeof(TensorOperations.$f)}, args...) = begin
            result = f.primal(map(x -> x.primal, args)...)
            pb(::Any) = (Mooncake.NoRData(), ntuple(_ -> Mooncake.NoRData(), length(args))...)
            return Mooncake.zero_fcodual(result), pb
        end
    end
end

# Special handling for tensorfree! and tensoralloc
Mooncake.is_primitive(::Type{<:DefaultCtx}, ::Type{<:Tuple{typeof(TensorOperations.tensorfree!), Vararg}}) = true
Mooncake.rrule!!(f::Mooncake.CoDual{typeof(TensorOperations.tensorfree!)}, args...) = begin
    pb(::Any) = (Mooncake.NoRData(), Mooncake.NoRData())
    return Mooncake.zero_fcodual(nothing), pb
end

Mooncake.is_primitive(::Type{<:DefaultCtx}, ::Type{<:Tuple{typeof(TensorOperations.tensoralloc), Vararg}}) = true
Mooncake.rrule!!(
    f::Mooncake.CoDual{typeof(TensorOperations.tensoralloc)},
    ttype::Mooncake.CoDual, structure::Mooncake.CoDual,
    istemp::Mooncake.CoDual, allocator::Mooncake.CoDual
) = begin
    # Always use Val(false) for istemp in AD context
    output = TensorOperations.tensoralloc(ttype.primal, structure.primal, Val(false), allocator.primal)
    pb(::Any) = (
        Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(),
        Mooncake.NoRData(), Mooncake.NoRData(),
    )
    return Mooncake.zero_fcodual(output), pb
end

# For the main tensor operations, we can use @from_rrule if the ChainRules are already defined
# You'll need to be specific about the types you want to support

# tensorscalar
@from_rrule DefaultCtx Tuple{typeof(tensorscalar), AbstractArray}

# tensoradd! - handle different arities
@from_rrule DefaultCtx Tuple{
    typeof(TensorOperations.tensoradd!),
    AbstractArray, AbstractArray, Index2Tuple, Bool,
    Number, Number,
}
@from_rrule DefaultCtx Tuple{
    typeof(TensorOperations.tensoradd!),
    AbstractArray, AbstractArray, Index2Tuple, Bool,
    Number, Number, Any,
}
@from_rrule DefaultCtx Tuple{
    typeof(TensorOperations.tensoradd!),
    AbstractArray, AbstractArray, Index2Tuple, Bool,
    Number, Number, Any, Any,
}

# tensorcontract!
@from_rrule DefaultCtx Tuple{
    typeof(TensorOperations.tensorcontract!),
    AbstractArray, AbstractArray, Index2Tuple, Bool,
    AbstractArray, Index2Tuple, Bool, Index2Tuple,
    Number, Number,
}
@from_rrule DefaultCtx Tuple{
    typeof(TensorOperations.tensorcontract!),
    AbstractArray, AbstractArray, Index2Tuple, Bool,
    AbstractArray, Index2Tuple, Bool, Index2Tuple,
    Number, Number, Any,
}
@from_rrule DefaultCtx Tuple{
    typeof(TensorOperations.tensorcontract!),
    AbstractArray, AbstractArray, Index2Tuple, Bool,
    AbstractArray, Index2Tuple, Bool, Index2Tuple,
    Number, Number, Any, Any,
}

# tensortrace!
@from_rrule DefaultCtx Tuple{
    typeof(tensortrace!),
    AbstractArray, AbstractArray, Index2Tuple, Index2Tuple,
    Bool, Number, Number,
}
@from_rrule DefaultCtx Tuple{
    typeof(tensortrace!),
    AbstractArray, AbstractArray, Index2Tuple, Index2Tuple,
    Bool, Number, Number, Any,
}
@from_rrule DefaultCtx Tuple{
    typeof(tensortrace!),
    AbstractArray, AbstractArray, Index2Tuple, Index2Tuple,
    Bool, Number, Number, Any, Any,
}

end
