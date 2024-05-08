using Random
"""
# Returns
- `X_train`: The training set features.
- `X_test`: The testing set features.

The function randomly shuffles the dataset and splits it according to the specified training proportion (`at`).
"""
function partitionTrainTest(data; at = 0.7,rng=Xoshiro())
    num_samples = size(data, 1)
    shuffled_indices = shuffle(rng,1:num_samples)
    split_index = floor(Int, at * num_samples)

    train_indices = view(shuffled_indices, 1:split_index)
    test_indices = view(shuffled_indices, (split_index + 1):num_samples)

    X_train = data[train_indices]
    X_test = data[test_indices]

    return X_train,  X_test
end


function numberPolynomials(n::Int64, d::Int64)
	x, y = max(d, n), min(d, n)
	return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end


# Macro for checking arguments
macro check_args(K, param, cond, desc=string(cond))
    quote
        if !($(esc(cond)))
            throw(
                ArgumentError(
                    string(
                        $(string(K)),
                        ": ",
                        $(string(param)),
                        " = ",
                        $(esc(param)),
                        " does not ",
                        "satisfy the constraint ",
                        $(string(desc)),
                        ".",
                    ),
                ),
            )
        end
    end
end


function vec_of_vecs(X::AbstractMatrix; obsdim::Union{Int,Nothing}=nothing)
    _obsdim = deprecated_obsdim(obsdim)
    if _obsdim == 1
        return RowVecs(X)
    elseif _obsdim == 2
        return ColVecs(X)
    else
        throw(ArgumentError("`obsdim` keyword argument should be 1 or 2"))
    end
end

"""
    ColVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` which interprets it as a vector-of-vectors, in
which each _column_ of `X` represents a single vector.

That is, by writing `x = ColVecs(X)`, you are saying "`x` is a vector-of-vectors, each of
which has length `size(X, 1)`. The total number of vectors is `size(X, 2)`."

Phrased differently, `ColVecs(X)` says that `X` should be interpreted as a vector
of horizontally-concatenated column-vectors, hence the name `ColVecs`.

```jldoctest
julia> X = randn(2, 5);

julia> x = ColVecs(X);

julia> length(x) == 5
true

julia> X[:, 3] == x[3]
true
```

`ColVecs` is related to [`RowVecs`](@ref) via transposition:
```jldoctest
julia> X = randn(2, 5);

julia> ColVecs(X) == RowVecs(X')
true
```
"""
struct ColVecs{T,TX<:AbstractMatrix{T},S} <: AbstractVector{S}
    X::TX
    function ColVecs(X::TX) where {T,TX<:AbstractMatrix{T}}
        S = typeof(view(X, :, 1))
        return new{T,TX,S}(X)
    end
end

Base.size(D::ColVecs) = (size(D.X, 2),)
Base.getindex(D::ColVecs, i::Int) = view(D.X, :, i)
Base.getindex(D::ColVecs, i::CartesianIndex{1}) = view(D.X, :, i)
Base.getindex(D::ColVecs, i) = ColVecs(view(D.X, :, i))
Base.setindex!(D::ColVecs, v::AbstractVector, i) = setindex!(D.X, v, :, i)

Base.vcat(a::ColVecs, b::ColVecs) = ColVecs(hcat(a.X, b.X))
Base.zero(x::ColVecs) = ColVecs(zero(x.X))

dim(x::ColVecs) = size(x.X, 1)

_to_colvecs(x::AbstractVector{<:Real}) = ColVecs(reshape(x, 1, :))



"""
    RowVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` which interprets it as a vector-of-vectors, in
which each _row_ of `X` represents a single vector.

That is, by writing `x = RowVecs(X)`, you are saying "`x` is a vector-of-vectors, each of
which has length `size(X, 2)`. The total number of vectors is `size(X, 1)`."

Phrased differently, `RowVecs(X)` says that `X` should be interpreted as a vector
of vertically-concatenated row-vectors, hence the name `RowVecs`.

Internally, the data continues to be represented as an `AbstractMatrix`, so using this type
does not introduce any kind of performance penalty.

```jldoctest
julia> X = randn(5, 2);

julia> x = RowVecs(X);

julia> length(x) == 5
true

julia> X[3, :] == x[3]
true
```

`RowVecs` is related to [`ColVecs`](@ref) via transposition:
```jldoctest
julia> X = randn(5, 2);

julia> RowVecs(X) == ColVecs(X')
true
```
"""
struct RowVecs{T,TX<:AbstractMatrix{T},S} <: AbstractVector{S}
    X::TX
    function RowVecs(X::TX) where {T,TX<:AbstractMatrix{T}}
        S = typeof(view(X, 1, :))
        return new{T,TX,S}(X)
    end
end

RowVecs(x::AbstractVector) = RowVecs(reshape(x, :, 1))

Base.size(D::RowVecs) = (size(D.X, 1),)
Base.getindex(D::RowVecs, i::Int) = view(D.X, i, :)
Base.getindex(D::RowVecs, i::CartesianIndex{1}) = view(D.X, i, :)
Base.getindex(D::RowVecs, i) = RowVecs(view(D.X, i, :))
Base.setindex!(D::RowVecs, v::AbstractVector, i) = setindex!(D.X, v, i, :)

Base.vcat(a::RowVecs, b::RowVecs) = RowVecs(vcat(a.X, b.X))
Base.zero(x::RowVecs) = RowVecs(zero(x.X))

dim(x::RowVecs) = size(x.X, 2)

dim(x) = 0 # This is the passes-by-default choice. For a proper check, implement `KernelFunctions.dim` for your datatype.
dim(x::AbstractVector) = dim(first(x))
dim(x::AbstractVector{<:AbstractVector{<:Real}}) = length(first(x))
dim(x::AbstractVector{<:Real}) = 1

function validate_inputs(x, y)
    if dim(x) != dim(y) # Passes by default if `dim` is not defined
        throw(
            DimensionMismatch(
                "dimensionality of x ($(dim(x))) is not equal to that of y ($(dim(y)))"
            ),
        )
    end
    return nothing
end

function validate_inplace_dims(K::AbstractMatrix, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    if size(K) != (length(x), length(y))
        throw(
            DimensionMismatch(
                "Size of the target matrix K ($(size(K))) not consistent with lengths of " *
                "inputs x ($(length(x))) and y ($(length(y)))",
            ),
        )
    end
end

function validate_inplace_dims(K::AbstractVector, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    n = length(x)
    if length(y) != n
        throw(
            DimensionMismatch(
                "Length of input x ($n) not consistent with length of input y " *
                "($(length(y))",
            ),
        )
    end
    if length(K) != n
        throw(
            DimensionMismatch(
                "Length of target vector K ($(length(K))) not consistent with length of " *
                "inputs ($n)",
            ),
        )
    end
end

function validate_inplace_dims(K::AbstractVecOrMat, x::AbstractVector)
    return validate_inplace_dims(K, x, x)
end



## Sorting
# https://discourse.julialang.org/t/how-to-sort-two-or-more-lists-at-once/12073/13
struct CoSorterElement{T1,T2}
    x::T1
    y::T2
end
struct CoSorter{T1,T2,S <: AbstractArray{T1},C <: AbstractArray{T2}} <: AbstractVector{CoSorterElement{T1,T2}}
    sortarray::S
    coarray::C
end

Base.size(c::CoSorter) = size(c.sortarray)
Base.getindex(c::CoSorter, i...) = 
    CoSorterElement(getindex(c.sortarray, i...), getindex(c.coarray, i...))
Base.setindex!(c::CoSorter, t::CoSorterElement, i...) = 
    (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c) 
Base.isless(a::CoSorterElement, b::CoSorterElement) = isless(a.x, b.x)
Base.Sort.defalg(v::C) where {T <: Union{Number,Missing},C <: CoSorter{T}} = 
    Base.DEFAULT_UNSTABLE

function sort_two_arrays!(x::Array, y::Array)
    T = CoSorter(x, y)
    sort!(T)
    x = T.sortarray
    y = T.coarray
end

function sort_two_arrays(x::Array, y::Array)
    xout = zeros(Float64, size(x))
    yout = zeros(Float64, size(y))
    T = CoSorter(deepcopy(x), deepcopy(y))
    sort!(T)
    xout = T.sortarray
    yout = T.coarray
    return xout, yout
end

