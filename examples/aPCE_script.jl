# ╔═╡ 83e8c530-773b-11ee-0a90-e9022fa50e8b
import Pkg
Pkg.activate("//iws-ls3-cifs.tik.uni-stuttgart.de/shared/users/ac125867/03_projects/32_aPC_julia/src")
using AddPackage
begin
    @add using MultivariateBases
    @add using DynamicPolynomials
    @add using FixedPolynomials
    @add using LinearAlgebra
    @add using Tullio
    @add using Distributions
    @add using Plots
    @add using Statistics
    @add using IterTools
    @add using Random
    @add using LazyGrids
    @add using Polynomials
    @add using SparseArrays
    @add using AMRVW
    @add using ToeplitzMatrices
    @add using FFTW
    @add using StaticArrays
    @add using Test
    @add using BenchmarkTools
    @add using ProfileCanvas
end

# ╔═╡ dd8f3396-5731-4dd1-981d-1e2a8ec6dcbb
html"""
<style>
	@media screen {
		main {
			margin: 0 auto;
			max-width: 3000px;
    		padding-left: max(100px, 5%);
    		padding-right: max(500px, 5%); 
            # 383px to accomodate TableOfContents(aside=true)
		}
	}
</style>
"""

# ╔═╡ 9542966c-be6d-4202-9b55-3fd28af49c5e
begin
    # Macro for checking arguments
    macro check_args(K, param, cond, desc = string(cond))
        return quote
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

    function deprecated_obsdim(obsdim::Union{Int, Nothing})
        _obsdim = if obsdim === nothing
            Base.depwarn(
                "implicit `obsdim=2` argument is deprecated and now has to be passed " *
                    "explicitly to specify that each column corresponds to one observation",
                :vec_of_vecs,
            )
            2
        else
            obsdim
        end
        return _obsdim
    end

    function vec_of_vecs(X::AbstractMatrix; obsdim::Union{Int, Nothing} = nothing)
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
    struct ColVecs{T, TX <: AbstractMatrix{T}, S} <: AbstractVector{S}
        X::TX
        function ColVecs(X::TX) where {T, TX <: AbstractMatrix{T}}
            S = typeof(view(X, :, 1))
            return new{T, TX, S}(X)
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
    struct RowVecs{T, TX <: AbstractMatrix{T}, S} <: AbstractVector{S}
        X::TX
        function RowVecs(X::TX) where {T, TX <: AbstractMatrix{T}}
            S = typeof(view(X, 1, :))
            return new{T, TX, S}(X)
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
        return if size(K) != (length(x), length(y))
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
        return if length(K) != n
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

end

# ╔═╡ 0579f0bd-5fc8-4c65-822f-9a9c6b0ace89
plotly()

# ╔═╡ 67289cdc-1676-451c-8bf1-7b4b9f184f48
function get_input(N, d, seed = 1)
    rng = Xoshiro(seed)
    if d == 2
        x = [rand(rng, Beta(2, 1), N) rand(rng, Uniform(0, 1), N)]
        return RowVecs(x)
    elseif d == 1
        x = rand(rng, Beta(2, 1), N)
        return RowVecs(x)
    else
        @error "Currently only d==1 and d==2 is implemented."
    end
end

# ╔═╡ 365ee76a-1fd9-439a-99e0-4ba80f3c9607
function PhysicalModelND(t, P::AbstractArray)
    # @debug "Assuming a nd case" P
    P = reduce(hcat, P)
    ModelResponse = (P[1]^2 + P[2] - 1.0) .^ 2 #+ P[1]^3 + 0.5 * P[1] * exp(P[2]) .- sqrt.(t) .* P[1]

    for i in 3:size(P, 1)
        ModelResponse .+= P[i]
    end

    return ModelResponse # SVector{length(ModelResponse)}(
end

# ╔═╡ 834cf93a-3aba-4b39-9705-8d85dc6ae752
function PhysicalModel1D(t, P)
    ModelResponse = @. (P[1]^2 + 0.0 - 1.0) .^ 2 #+ P[1]^3 + 0.5 * P[1] * exp(0.0) .- sqrt.(t) .* P[1]

    for i in 3:size(P, 1)
        ModelResponse .+= P[i]
    end

    return ModelResponse # SVector{length(ModelResponse)}(
end

# ╔═╡ f4a00a4b-912e-4039-ac86-0ae8ff4854a3
function numberPolynomials(n::Int64, d::Int64)
    x, y = max(d, n), min(d, n)
    return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end

# ╔═╡ 4f3ed422-6ce1-43e4-9d8f-6bbc35a9f763
function reverse_columns!(x)
    for row in axes(x, 1)
        x[row, :] = reverse(x[row, :])
    end
    return
end

# ╔═╡ 062107a0-de05-4ecd-a8e0-ea3779f2fb42
function aPC_MultivariatePolynomialDegrees(N, d)
    # Input:
    # N- Number of uncertain parameters
    # d - Degree of polynomial expansion
    # Output:
    # PolynomialDegree - Multivariate Polynomial Degrees
    # Total number of terms
    P = numberPolynomials(N, d)
    # Possible Degrees
    UniqueDegreeCombinations = zeros((d + 1)^N, d)
    PossibleDegrees = [collect(0:d) for _ in N:-1:1]
    if N == 1
        UniqueDegreeCombinations = PossibleDegrees[1]
    else
        tmp = cat(collect(ndgrid_array(reverse(PossibleDegrees)...))...; dims = 3)
        UniqueDegreeCombinations = reshape(tmp, :, N)
    end
    # Possible degree computation
    DegreeWeight = zeros(1, size(UniqueDegreeCombinations, 1))
    for i in 1:1:size(UniqueDegreeCombinations, 1)
        DegreeWeight[i] = 0.0
        for j in 1:1:N
            DegreeWeight[i] = DegreeWeight[i] + UniqueDegreeCombinations[i, j]
        end
    end
    # Sorting of possible degree
    id = sortperm(DegreeWeight; dims = 2)[:]
    SortDegreeCombinations = UniqueDegreeCombinations[id, :]
    # Multivariate Polynomial Degrees
    reverse_columns!(SortDegreeCombinations)
    return SortDegreeCombinations[1:P, :]
end

# ╔═╡ e2ed7021-a13c-4113-b3d3-618520c6c7c8
function aPC_OrthonormalBasis(Data, Degree)
    d = Degree  #Degree of polinomial expansion
    dd = d + 1 #Degree of polinomial for roots defenition
    L_norm = 1  # L-norm for polnomial normalization
    NumberOfDataPoints = length(Data)
    # @info "Construction of Arbitrary Polynomial Basis" --> We do d+2 x d+2, to have one order higher polynomials to get the gaussian quadrature poitns in the end!!!

    MeanOfData = mean(Data)
    VarOfData = var(Data)
    Data = Data ./ MeanOfData
    m = zeros(2 * dd + 2)
    for i in 0:(2 * dd + 1)
        m[i + 1] = sum(Data .^ i) / NumberOfDataPoints  # Raw Moments
    end
    poly = zeros(dd + 1, dd + 1)
    MHankel = zeros(dd + 1, dd + 1) # Allocate once for all :)
    for degree in 0:dd
        Hankel = @views MHankel[1:(degree + 1), 1:(degree + 1)]
        Hankel .= 0.0
        fr = copy(Hankel)
        # fr1 = copy(fr)
        Vc = zeros(degree + 1)
        PolyCoeff_NonNorm = copy(Hankel)

        for i in 0:degree
            for j in 0:degree
                if i < degree
                    Hankel[i + 1, j + 1] = m[i + j + 1] # put in the moment
                elseif (i == degree) && (j < degree)
                    Hankel[i + 1, j + 1] = 0.0
                elseif (i == degree) && (j == degree)
                    Hankel[i + 1, j + 1] = 1.0
                end
            end
            # fr1 = copy(Hankel); # Control Hankel only considering the raw moments without division by max(abs) in each row
            Hankel[i + 1, :] = Hankel[i + 1, :] / maximum(abs.(Hankel[i + 1, :]))
        end

        for i in 0:degree
            if (i < degree)
                Vc[i + 1] = 0
            elseif (i == degree)
                Vc[i + 1] = 1
            end
        end

        # inv_Mm = pinv(Hankel, atol=1e-6)
        # display(Hankel)
        # dt = copy(Vc)
        # fr .= Hankel
        Vp = Hankel \ Vc
        PolyCoeff_NonNorm[degree + 1, 1:(degree + 1)] .= Vp
        if (100 * abs(sum(abs.(Hankel * PolyCoeff_NonNorm[degree + 1, 1:(degree + 1)])) - sum(abs.(Vc))) > 0.5)
            @warn "Computational error of the linear solver is too high"
        end


        #Normalization of polynomial coefficients
        P_norm = 0
        for i in 1:NumberOfDataPoints
            Poly = 0
            for k in 0:degree
                Poly += PolyCoeff_NonNorm[degree + 1, k + 1] * Data[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end

        for k in 0:degree
            poly[degree + 1, k + 1] = PolyCoeff_NonNorm[degree + 1, k + 1] / sqrt(P_norm)
        end


    end

    # Backward linear transformation to the data space
    Data = Data * MeanOfData
    for k in 1:size(poly, 2)
        poly[:, k] = poly[:, k] ./ (MeanOfData^(k - 1))
    end

    #%% Data-driven Arbitrary Orthonormal Polynomial Basis
    OrthonormalBasis = poly
    return OrthonormalBasis # SMatrix{dd+1,dd+1}(
end

# ╔═╡ e5544ae4-1934-4867-9e59-3bb01928341e
mutable struct aPC{T <: Real}
    InputDistribution::RowVecs{T} # in [ d x N-samples]
    input_dimensions::Int64
    ExpansionDegree::Int64
    NumberOfTerms::Int64
    MultivariatePolynomialDegrees::AbstractArray{Int64}
    OrthonormalRepresentation::Bool
    OrthonormalBasis::AbstractArray{T}
    # NumberOfOutputs::Int64
    ExpansionCoefficients::AbstractVector{T}

    # Constructor
    function aPC(
            InputDistribution::RowVecs{T},
            ExpansionDegree::Int64,
            OrthonormalRepresentation::Bool = true
        ) where {T}
        input_dimensions = size(InputDistribution[1], 1)
        # @info "" size(InputDistribution[1])
        MultivariatePolynomialDegrees = aPC_MultivariatePolynomialDegrees(input_dimensions, ExpansionDegree)
        NumberOfTerms = numberPolynomials(ExpansionDegree, input_dimensions)

        OrthonormalBasis = zeros(ExpansionDegree + 2, ExpansionDegree + 2, input_dimensions)
        for i in 1:input_dimensions
            tmp = aPC_OrthonormalBasis(getindex.(InputDistribution, i), ExpansionDegree)
            OrthonormalBasis[:, :, i] .= tmp
        end
        # OrthonormalBasis = tmp

        # display(OrthonormalBasis)
        NumberOfOutputs = 1 # Allocate 100
        ExpansionCoefficients = zeros(T, NumberOfTerms)

        return new{T}(
            InputDistribution,
            input_dimensions,
            ExpansionDegree,
            NumberOfTerms,
            MultivariatePolynomialDegrees,
            OrthonormalRepresentation,
            OrthonormalBasis,
            # NumberOfOutputs,
            ExpansionCoefficients,
        )
    end
end

# ╔═╡ c1d9ab51-9bbb-4f54-9377-1ea79357b09e
function GaussianCollocation(apc::aPC{T}; Strategy = :FT)::RowVecs{T} where {T <: Real}
    AvailableCollocationPoints = zeros(ComplexF64, apc.input_dimensions, apc.ExpansionDegree + 1)
    for d in 1:apc.input_dimensions
        poly = apc.OrthonormalBasis[:, :, d]
        AvailableCollocationPoints[d, :] = Polynomials.roots(Polynomials.Polynomial(poly[apc.ExpansionDegree + 2, :]))
        # AvailableCollocationPoints[d,:]  = AMRVW.roots(poly[apc.ExpansionDegree+2,:])

    end
    AvailableCollocationPoints = reverse(AvailableCollocationPoints)

    PointsVector = 1:(apc.ExpansionDegree + 1) |> collect
    UniqueCombinations = stack(reduce(vcat, collect(Iterators.product([PointsVector for i in 1:apc.input_dimensions]...))))' |> collect

    DigitalPointsWeight = zeros(size(UniqueCombinations, 1))

    for (i, r) in enumerate(eachrow(UniqueCombinations))
        DigitalPointsWeight[i] = sum(r)
    end


    index_SDPW = sortperm(reshape(DigitalPointsWeight, (:, 1)); dims = 1)[:]
    # display(index_SDPW)
    SortUniqueCombinations = UniqueCombinations[index_SDPW, :]

    TrainingInput = SortUniqueCombinations

    # @info "Generated Gaussian collocation input"
    return  RowVecs(Array{T}(view(Float64.(TrainingInput), :, reverse(1:size(TrainingInput, 2)))))
end


# ╔═╡ 755ee209-17d3-4cd0-a15b-b8ff0dba97ff
# function aPC_PsiPolynomialMatrix(apc::aPC{T},TrainingInput)::AbstractMatrix{T} where {T<:Real}
# 	d = apc.ExpansionDegree; #Degree of polinomial expansion
# 	dd = d+1 #Degree of polinomial for roots defenition
# 	NumberOfTerms = apc.NumberOfTerms
# 	NCpoints =  size(TrainingInput,1)
# 	@debug "" NCpoints
# 	Psi =  ones(NumberOfTerms,NCpoints)
# 	PolynomialDegree = apc.MultivariatePolynomialDegrees
# 	@show PolynomialDegree
# 	@debug "" apc.OrthonormalBasis
# 	for i=1:NumberOfTerms
# 	    for j=1:NCpoints
# 			Psi[i,j] = 1.0
# 	        for ii=1:apc.input_dimensions # They get multiplied.
# 					pd = PolynomialDegree[i,d]+1
# 					p = Polynomials.Polynomial(apc.OrthonormalBasis[pd,1:end-2,d])
# 					x = getindex(TrainingInput[j],ii)
# 				@debug "" TrainingInput[j]
# 		            Psi[i,j]=Psi[i,j]*p(x)
# 	        end
# 	    end
# 	end
# 	return Psi
# end

# ╔═╡ 2179ccb3-e486-402d-a131-9148e12ad1c2
function aPC_PsiPolynomialMatrix(apc::aPC{T}, TrainingInput) where {T <: Real}
    NumberOfTerms, InputDimensions = size(apc.MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = ones(T, NumberOfTerms, NCpoints)
    @debug "" TrainingInput
    for i in 1:NumberOfTerms  # For each term in the polynomial expansion
        for j in 1:NCpoints  # For each input sample
            product = 1.0  # Initialize the product for this term and sample
            for ii in 1:InputDimensions  # For each dimension of the input
                degree = apc.MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
                coeffs = apc.OrthonormalBasis[degree, 1:degree, ii]  # Extract the coefficients for the polynomial
                p = Polynomials.Polynomial(coeffs)  # Create the polynomial
                x = reduce(hcat, TrainingInput)[ii, j]
                # @show degree coeffs p x p(x)

                product *= p(x)  # Evaluate the polynomial at x and multiply
            end
            Psi[i, j] = product  # Assign the product to Psi matrix
        end
    end

    return Psi
end

# ╔═╡ 59171b1f-8c1b-4865-af39-ddc795ecc0b3
@testset verbose = true showtiming = true "All tests" begin
    @testset "aPC_OrthonormalBasis" begin
        @test aPC_OrthonormalBasis([1 / sqrt(3), -1 / sqrt(3), 1.0], 1) ≈ SMatrix{3, 3}(1, -0.5, -2.0, 0.0, 1.5, -1.5, 0.0, 0.0, 4.5)
    end

    @testset "aPC_MultivariatePolynomialDegrees" begin
        @test aPC_MultivariatePolynomialDegrees(2, 1) == SMatrix{3, 2}(0, 0, 1, 0, 1, 0)
        @test aPC_MultivariatePolynomialDegrees(2, 2) == SMatrix{6, 2}(0, 0, 1, 0, 1, 2, 0, 1, 0, 2, 1, 0)
    end

end


# ╔═╡ 5f8640cf-99f6-40cc-ac14-2101fe0dd5d9
function train!(apc::aPC{T}, TrainingInput::RowVecs, TrainingOutput::RowVecs) where {T <: Real}
    # @info "Training the arbitrary Polynomial Chaos"
    # @warn "" size(TrainingInput)
    Psi = aPC_PsiPolynomialMatrix(apc, TrainingInput)'
    to = reduce(vcat, TrainingOutput)
    @debug "" size(Psi) Psi to size(to) TrainingOutput
    # 	# if  apc.input_dimensions == 1
    # Psi_inv = pinv(Psi)

    # 	# @debug ""  Psi_inv TrainingOutput
    # apc.ExpansionCoefficients = Psi_inv*to
    # display(apc.ExpansionCoefficients )
    # else
    Psi_inv = pinv(Psi)
    # @debug ""  Psi_inv to
    apc.ExpansionCoefficients = Psi_inv * to
    # apc.ExpansionCoefficients = Psi\to
    # end
    return nothing
end

# ╔═╡ e06776cb-cf7c-4693-b1b1-d6577d72c559
# function predict(apc::aPC{T},PredictionInput::RowVecs{S}) where {S<:Real,T<:Real}
# 	# @warn "" size(PredictionInput)
# 	# if  apc.input_dimensions == 1
# 		# Psi = aPC_PsiPolynomialMatrix(apc,PredictionInput')'
# 		# PredictionOutput = vec(Psi.*apc.ExpansionCoefficients')
# 	# 	return PredictionOutput
# 	# else
# 	Psi = aPC_PsiPolynomialMatrix(apc,PredictionInput)
# 	# @debug "" Psi
# 	PredictionOutput = zeros(size(PredictionInput,1))
# 	for (i,col) in enumerate(eachcol(Psi))
# 		PredictionOutput[i] =  col'*apc.ExpansionCoefficients
# 	end
# 	@debug "" size(Psi) size(apc.ExpansionCoefficients)
# 	# display(Psi)

# 	# PredictionOutput = Psi*apc.ExpansionCoefficients
# 	return PredictionOutput
# 	# end
# end

# ╔═╡ 331760b1-3346-4ed9-b9b2-e14b936ec3c6
function predict(apc::aPC{T}, PredictionInput) where {T <: Real}
    println("=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ...")

    # Assuming aPC_PsiPolynomialMatrix is correctly implemented in Julia as discussed before
    Psi = aPC_PsiPolynomialMatrix(apc, PredictionInput)'
    # Initialize the prediction output matrix
    # @debug "" Psi apc.ExpansionCoefficients
    # PredictionOutput = zeros(T, size(PredictionInput, 1))
    # Perform prediction using the expansion coefficients
    # PredictionOutput = Psi * apc.ExpansionCoefficients
    PredictionOutput = [dot(apc.ExpansionCoefficients, row) for row in eachrow(Psi)]
    return PredictionOutput
end

# ╔═╡ d1efcbbd-7717-4dc3-a293-7fec3c86e675
function UQ(apc::aPC{T}) where {T <: Float64}
    lc = Array{T}(copy(apc.ExpansionCoefficients))
    OutputMean = Vector{Float64}(lc[1, :])
    OutputVar = Vector{Float64}(sum(lc[2:end, :] .^ 2; dims = 1)[:])
    return (OutputMean = OutputMean, OutputVar = OutputVar)
end

# ╔═╡ ac63c5ec-e59d-451a-b8f0-eefcae2f5bbb
# let
# 	err = []
# 	ts = []
# 	ps = []
# 	degress = 2:3:10
# 	N = 100
# 	d = 2

# 	x =  get_input(N,d,5)

# 	m(x) = PhysicalModel1D(1,x)
# 	last_pred = undef
# 	gTrainingInput = []

# 	for degree in degress
# 		@info "" degree
# 		t = @elapsed begin
# 	apc_instance = aPC(x, degree);
# 	@info "" apc_instance
# 	TrainingInput =  GaussianCollocation(apc_instance)
# 			gTrainingInput = TrainingInput
# 	# apc_instance = aPC(x, degree);
# 	# @info "" apc_instance
# 	# TrainingInput =  GaussianCollocation(apc_instance)
# 	# @debug "" TrainingInput
# 	TrainingOutput = []
# 	for i=1:apc_instance.NumberOfTerms
# 		if d == 2
#     		push!(TrainingOutput,m(TrainingInput[i,:]))
# 		elseif d==1
#     		push!(TrainingOutput,m(TrainingInput[i,:][1]))
# 		end
# 	end
# 	TrainingOutput = reduce(hcat,TrainingOutput)'|>collect
# 	# @debug "" size(TrainingOutput) TrainingOutput
# 	train!(apc_instance,TrainingInput,TrainingOutput)
# 	# @info "" apc_instance
# 	# @debug "" UQ(apc_instance)

# 			# @debug "" xcp
# 	pred = predict(apc_instance,TrainingInput)
# 	# display(pred)

# 	last_pred = yy->predict(apc_instance,yy)

# 	mypred = [last_pred(x)|>first for x in eachrow(TrainingInput)]

# 	expect = m.(eachrow(TrainingInput))

# 	@debug "" size(pred) size(expect)
# 	# display(expect)
# 	# @debug "" size(expect) size(pred)
# 	# push!(err,mean(abs.(pred.-expect).^2))
# 	# 		@debug expect.-pred
# 		end
# 	# push!(ts,t)

# 	# xtest = gTrainingInput
# 	# p3 = Plots.scatter(xtest[:,1], m.(xtest')',alpha=0.5)
# 	# preds = [last_pred(x)|>first for x in eachrow(xtest)]
# 	# @show preds
# 	# Plots.scatter!(p3,xtest[:,1], preds,alpha=0.5)

# 	# 	push!(ps,p3)
# 	end
# 	# p1 = Plots.scatter(degress, log.(ts))
# 	# p2 = Plots.scatter(degress, log.(err))


# 	# pn = Plots.plot(p1,p2)
# 	# pn2 = Plots.plot(ps...)
# 	# Plots.plot(pn,pn2, layout =@layout [a ; b])
# 	# Plots.scatter(log.((abs.(pred[1,:].-expect[1,:]))))
# end

# ╔═╡ 939c9b57-bc84-4c10-b330-d8fd1a6f073e
let
    err = []
    ts = []
    ps = []
    # degress = 1:1:5
    N = 1000
    d = 2

    x = get_input(N, d, 1)
    # @debug "" x
    # m(x) = PhysicalModel1D(1,x)
    last_pred = undef

    true_output = [ PhysicalModelND(1, ix) for ix in x]

    degree = 2
    # t = @elapsed begin
    apc_instance = aPC(x, degree)
    # @info "" apc_instance
    TrainingInput = GaussianCollocation(apc_instance)


    TrainingOutput = []
    for i in 1:size(TrainingInput, 1)
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

    @info "" TrainingInput TrainingOutput
    # @info "" apc_instance
    train!(apc_instance, TrainingInput, TrainingOutput)
    @info "" apc_instance

    # # @debug "" UQ(apc_instance)
    # # xcp = [0.422117914428017	0.610608061392019	0.780825721677619]]


    pred = predict(apc_instance, x)
    # display(pred)
    # last_pred = yy-> predict(apc_instance,[yy])

    # Plots.plot(x,TrainingOutput)
    p1 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], true_output, ms = 1)
    p2 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred, ms = 1)
    p3 = Plots.scatter(reduce(hcat, x)[1, :], reduce(hcat, x)[2, :], pred .- true_output, ms = 1)

    # Plots.scatter!(p2,reduce(hcat,TrainingInput)[2,:],)
    # expect = RowVecs(PhysicalModel1D.(1,TrainingInput))
    # display(expect)
    Plots.plot(p1, p2, p3)

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

# ╔═╡ 8a54a0a0-2a3c-4757-b908-9d2765074482
# function testmodel(x, NumberOfOutputs=3)
# 	function PhysicalModel(t, P)
#     if length(P) == 1
#         P[2] = 0.0
#     end

#     ModelResponse = (P[1]^2 + P[2] - 1.0).^2 + P[1]^3 + 0.5 * P[1] * exp(P[2]) .- sqrt.(t) .* P[1]

#     for i = 3:length(P)
#         ModelResponse .+= P[i]
#     end

#     return ModelResponse
# end

# 	TrainingOutput = []
# 		m(x) = PhysicalModel(1:NumberOfOutputs,x)
# 		for i=1:apc_instance.input_dimensions
# 	    	push!(TrainingOutput,m(TrainingInput[i,:]))
# 		end
# 		TrainingOutput = reduce(hcat,TrainingOutput)'|>collect
# 	return TrainingOutput
# end

# ╔═╡ 417151da-08e0-49d7-bfad-35a6e77d7715
# function create_apc_predictor(x,expansion_degree, model)
# 	apc_instance = aPC(x, expansion_degree);
# 	TrainingInput =  GaussianCollocation(apc_instance)
# 	TrainingOutput = model(TrainingInput)
# 	train!(apc_instance,TrainingInput,TrainingOutput)
# 	return x_eval-> begin
# 		predict(apc_instance,x_eval)
# 	end
# end

# ╔═╡ 2f9757a1-71e7-44c2-bcb4-db5475d3b130
# f = create_apc_predictor(x,7,testmodel)

# ╔═╡ 1b0be5e2-c131-4a34-97df-b4e44c4862a0
# x

# ╔═╡ bbcc2b04-f95b-495c-8216-618b81726c7e
# f(x)
