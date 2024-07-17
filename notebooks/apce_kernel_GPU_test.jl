### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 6ddcc330-436f-11ef-28f8-b76e24aed2b4
begin
	using CUDA
	using KernelAbstractions
	using TensorOperations
	using DispatchDoctor
	using Tullio
	using Zygote
	using Chairmarks
	using Random
	using DifferentiationInterface
	using Einsum
	using TimerOutputs
	using Statistics
	using Enzyme
	using ForwardDiff
	using ReverseDiff
	using ChainRulesCore
	using cuTENSOR
	using OMEinsum
	using Bumper
	import .EnzymeRules
	using .EnzymeRules
end

# ╔═╡ c95eb473-bced-4fa4-a686-c07a28992e48
ChainRulesCore.debug_mode() = true

# ╔═╡ af1a1731-3e1b-4ebe-a527-3103a5bda87a
@be let
	to = TimerOutput()
	i = 100
	j = 8
	k = 3
	Ψ = cu(rand(rng,Float32,i,j))
	α = cu(rand(rng,Float32,i,k))
	function obj(x)
		@tensor value[k, j] := Ψ[i, k] * α[i, j]
		return mean(value)
	end
	backend = DifferentiationInterface.AutoZygote()
	# @be DifferentiationInterface.gradient(x->obj(x),backend,α)
	A = DifferentiationInterface.gradient(obj,backend,α)
end

# ╔═╡ 623cd8e8-bf2e-4a96-95ff-2ce5ee928737
@be let
	to = TimerOutput()
	i = 100
	j = 8
	k = 3
	Ψ = cu(rand(rng,Float32,i,j))
	α = cu(rand(rng,Float32,i,k))
	function obj(x)
		value = outer_product_kernel(Ψ,x)
		return mean(value)
	end
	backend = DifferentiationInterface.AutoZygote()
	# @be DifferentiationInterface.gradient(x->obj(x),backend,α)
	A = DifferentiationInterface.gradient(obj,backend,α)
end

# ╔═╡ cdfcc712-c291-4db4-acde-fe0d14f8cb25
# let
# 	function f(y, x)
#     y .= x.^2
#     return sum(y)
# 	end
	
# 	function EnzymeRules.forward(func::Const{typeof(f)}, ::Type{<:Duplicated}, y::Duplicated, x::Duplicated)
#     println("Using custom rule!")
#     ret = func.val(y.val, x.val)
#     y.dval .= 2 .* x.val .* x.dval
#     return Duplicated(ret, sum(y.dval))
# end
# 	x  = [3.0, 1.0]
# 	dx = [1.0, 0.0]
# 	y  = [0.0, 0.0]
# 	dy = [0.0, 0.0]

# g(y, x) = f(y, x)^2 # function to differentiate

# @show autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of g w.r.t. x[1]
# @show dy; # derivative of y w.r.t. x[1] when g is run
# end

# ╔═╡ b12a9835-a1fb-4b20-8992-85a45f347d5f
# let
# 	to = TimerOutput()
# 	i = 100
# 	j = 8
# 	k = 3
# 	Ψ = cu(rand(rng,Float32,i,j))
# 	dΨ = copy(Ψ)
# 	α = cu(rand(rng,Float32,i,k))
# 	dα = copy(α)
# 	function obj(x)
# 		out = outer_product_kernel(Ψ,x)
# 	return mean(Array(out)[:])
# 	end
# 	# backend = DifferentiationInterface.AutoEnzyme()
# 	# DifferentiationInterface.gradient(x->obj(x),backend,α)
# 	autodiff(Forward, outer_product_kernel, Duplicated(Ψ, dΨ), Duplicated(α, dα))
# end

# ╔═╡ 2ba00792-7be0-4dd1-ba76-cec8796e6d47
begin
@stable my_outer_product(Ψ, expansion_coefficients) = my_outer_product(Base.promote_op(*, eltype(Ψ), eltype(expansion_coefficients)), Ψ, expansion_coefficients)
@stable function my_outer_product(::Type{T}, Ψ, expansion_coefficients) where {T}
    # Get the dimensions of the input matrices
    (i_dim, k_dim) = size(Ψ)
    (i_dim_exp, j_dim) = size(expansion_coefficients)
    # if i_dim != i_dim_exp
    #     throw(DimensionMismatch("The dimensions of Ψ and expansion_coefficients are not compatible"))
    # end
    result = zeros(T, k_dim, j_dim)
    @inbounds for i in 1:i_dim
        for k in 1:k_dim
            @simd for j in 1:j_dim
                result[k, j] += Ψ[i, k] * expansion_coefficients[i, j]
            end
        end
    end
    return result
end

end

# ╔═╡ 277bf495-88e0-4970-ae3f-0e9dd488beff
begin
	
@kernel function outer_product_kernel!(output, Ψ, expansion_coefficients)
    i, j = @index(Global, NTuple)
        for k in 1:size(output, 1)
            @inbounds output[k, j] += Ψ[i, k] * expansion_coefficients[i, j]
        end
end

# Creating a wrapper kernel for launching with error checks
function outer_product!(output, Ψ, expansion_coefficients)
    backend = KernelAbstractions.get_backend(Ψ)
    kernel! = outer_product_kernel!(backend)
    kernel!(output, Ψ, expansion_coefficients, ndrange=size(expansion_coefficients))
end

@inline function outer_product_kernel(Ψ::AbstractArray{T},α::AbstractArray{S}) where {T<:Real, S<:Real}
	(i_dim, k_dim) = size(Ψ)
    (i_dim_exp, j_dim) = size(α)
	backend =get_backend(Ψ) # or KernelAbstractions.CUDA() for GPU
	KernelAbstractions.synchronize(backend)
	out = KernelAbstractions.zeros(backend, S,k_dim,j_dim)
	outer_product!(out, Ψ, α)
	return out
end

	
function ChainRulesCore.rrule(::typeof(outer_product_kernel), Ψ, α::AbstractArray{T}) where {T<:Real}
	Ψ = Array(Ψ)
	α = Array( α)
    result = my_outer_product(Ψ, α)
    function pullback(Δresult)
        (i_dim, k_dim) = size(Ψ)
        (_, j_dim) = size(α)
		ΔΨ = zeros(T,i_dim,k_dim)
		Δα = zeros(T,i_dim,j_dim)
        @inbounds for i in 1:i_dim
            for k in 1:k_dim
         		@simd for j in 1:j_dim
                    ΔΨ[i, k] += Δresult[k, j] * α[i, j]
                    Δα[i, j] += Δresult[k, j] * Ψ[i, k]
                end
            end
        end
        return (NoTangent(), ΔΨ, Δα)
	end
	return result, pullback
    end
	
# @inline function outer_product_derivatives!(ΔΨ,Δα, Ψ, α,Δresult )
#     backend = KernelAbstractions.get_backend(Ψ)
# 	KernelAbstractions.synchronize(backend)
	
#     kernel! = outer_product_kernel_derivatives!(backend)
# 	KernelAbstractions.synchronize(backend)
	
#     kernel!(ΔΨ, Δα, Ψ, α,cu(Δresult),ndrange=size(α))
# end


# 	@kernel function outer_product_kernel_derivatives!(ΔΨ,Δα, Ψ, α,Δresult)
#     i, j = @index(Global, NTuple)
#         for k in 1:size(ΔΨ, 1)
# 			   ΔΨ[i, k] += Δresult[k, j] * α[i, j]
#                Δα[i, j] += Δresult[k, j] * Ψ[i, k]
#        end
# 	end
	
# function ChainRulesCore.rrule(::typeof(outer_product_kernel), Ψ, α)
# 	backend = get_backend(Ψ)
#     result = outer_product_kernel(Ψ, α)
#     function pullback(Δresult)
# 		Δresult = cu(Δresult)
#         (i_dim, k_dim) = size(Ψ)
#         (_, j_dim) = size(α)
# 		ΔΨ = CUDA.zeros(i_dim,k_dim)
# 		Δα = CUDA.zeros(i_dim,j_dim)
#         outer_product_derivatives!(ΔΨ,Δα, Ψ, α,Δresult )
#     	# KernelAbstractions.synchronize(backend)
#         return (NoTangent(), ΔΨ, Δα)
# 	end
# 	return result, pullback
#     end
 

end

# ╔═╡ 210a3ec9-2c45-4520-9cd2-674a3d6f9fad
rng = Xoshiro(123)

# ╔═╡ b23aa282-2c59-43ae-96f9-61a268388747
let
	to = TimerOutput()
	i = 5000
	j = 8
	k = 3
	Ψ = rand(rng,i,j)
	α = rand(rng,i,k)
	for _ in 1:20
	
	@timeit to "my_outer_cpu" my_outer_product(Ψ,α)
	@timeit to "tensoropt_cpu"  let
		@tensoropt value[k, j] := Ψ[i, k] * α[i, j]
	end

		@timeit to "butensort_cpu"  let
		@butensor value[k, j] := Ψ[i, k] * α[i, j]
	end
		
	@timeit to "tensor_cpu"  let
		@tensor value[k, j] := Ψ[i, k] * α[i, j]
	end
			@timeit to "tullio_cpu"  let
		@tullio value[k, j] := Ψ[i, k] * α[i, j]
		value
	end

			@timeit to "ein_cpu"  let
		@ein value[k, j] := Ψ[i, k] * α[i, j]
	end
	
			@timeit to "einsum_cpu"  let
	@einsum value[k, j] := Ψ[i, k] * α[i, j]
	end
	end
	display(to)
end

# ╔═╡ 5aeac385-f1a1-4a0f-8f50-5f1fc8fdddbd
let
	
	to = TimerOutput()
	i = 10000
	j = 8
	k = 3
	Ψ = rand(rng,i,j)
	α = rand(rng,i,k)
	Ψ2 = cu(Ψ)
	α2 = cu(α)
	for _ in 1:1000
		@timeit to "my_outer_cpu" my_outer_product(Ψ,α)
		Ψ2 = cu(Ψ)
		α2 = cu(α)
		@timeit to "outer_gpu_product"  let
			output = outer_product_kernel(Ψ2,α2)
		end
			@timeit to "tensor_gpu"  let
			@cutensor  value[k, j] := Ψ2[i, k] * α2[i, j]
		end
		
		@timeit to "ein_gpu"  let
			@ein value[k, j] := Ψ2[i, k] * α2[i, j]
		end
		# 	try
		# @timeit to "tullio_gpu"  let
		# 	@tullio value[k, j] := Ψ2[i, k] * α2[i, j]
		# end
		# 	catch
		# 	end
	end
	display(to)
end

# ╔═╡ 4abbcdff-2d44-4a93-beba-6bc89ca20454
# let
# 		i = 10000
# 	j = 8
# 	k = 3
# 		Ψ = rand(rng,i,j)
# 	α = rand(rng,i,k)
# 	Ψ2 = cu(Ψ)
# 	α2 = cu(α)
# output = outer_product_kernel(Ψ2,α2)

# ╔═╡ 2b49ab93-593c-4633-8548-426d6de50efb
# 	let
# 		to = TimerOutput()
# 		i = 5000
# 		j = 5
# 		k = 3
		
# 		Ψ = rand(rng,i,j) |>cu
# 		α = rand(rng,i,k) |>cu
	
	
	
# 	display(Array(output) )

# 	# display(@tullio value[k, j] := Ψ[i, k] * α[i, j])
		
# end

# ╔═╡ 5ec804ae-9e8e-4cae-9f71-cb9347405d5b
# 	my_outer_product_kernel(Ψ,α)
	# 		# @timeit to "my_outer_cpu" my_outer_product(Ψ,α)
		
	# 	# @timeit to "tensoropt_cpu"  let
	# 	# 	@tensoropt value[k, j] := Ψ[i, k] * α[i, j]
	# 	# end
		
	# 	# 		@timeit to "tullio_cpu"  let
	# 	# 	@tullio value[k, j] := Ψ[i, k] * α[i, j]
	# 	# 	value
	# 	# end
		
	# 	# 		@timeit to "einsum_cpu"  let
	# 	# @einsum value[k, j] := Ψ[i, k] * α[i, j]
	# 	# end
	# 	display(to)
	# end

# ╔═╡ a472ad1f-122e-4b5c-990a-23330721e21f
let
	to = TimerOutput()
	i = 100
	j = 8
	k = 3
	Ψ = cu(rand(rng,Float32,i,j))
	α = cu(rand(rng,Float32,i,k))
	outer_product_kernel(Ψ,α)
end

# ╔═╡ 69554245-386a-4c9d-9e48-53115b58f34f
function obj(Ψ,x)
out = outer_product_kernel(Ψ,x)
	return mean(Array(out)[:])
end

# ╔═╡ ced5c143-1227-49c8-9cea-3ad680745cb6
let
	to = TimerOutput()
	i = 100
	j = 8
	k = 3
	Ψ = rand(rng,Float32,i,j)
	α = rand(rng,Float32,i,k)
	function obj(x)
		@tensor value[k, j] := Ψ[i, k] * x[i, j]
	return mean(Array(value)[:])
	end
	backend = DifferentiationInterface.AutoZygote()
	@be DifferentiationInterface.gradient(x->obj(x),backend,α)
	DifferentiationInterface.gradient(x->obj(x),backend,α)
end

# ╔═╡ 72cef223-ef79-4f21-b5c0-d3d3eb097ace
# unction rrule(::typeof(*), A::AbstractMatrix, B::AbstractMatrix)
#     function times_pullback(ȳ)
#         dA = ȳ * B'
#         dB = A' * ȳ
#         return NoTangent(), dA, dB
#     end
#     return A * B, times_pullback 
# end

# ╔═╡ 638b7fa7-be3b-4c41-abf6-19d26035d47a
function ChainRulesCore.rrule(::typeof(my_outer_product), Ψ, α)
    result = my_outer_product(Ψ, α)
    function pullback(Δresult)
        (i_dim, k_dim) = size(Ψ)
        (_, j_dim) = size(α)
		ΔΨ = zeros(i_dim,k_dim)
		Δα = zeros(i_dim,j_dim)
        @inbounds for i in 1:i_dim
            for k in 1:k_dim
         		@simd for j in 1:j_dim
                    ΔΨ[i, k] += Δresult[k, j] * α[i, j]
                    Δα[i, j] += Δresult[k, j] * Ψ[i, k]
                end
            end
        end
        return (NoTangent(), ΔΨ, Δα)
	end
	return result, pullback
    end

# ╔═╡ 08d56edc-827a-4429-8e64-7eee13c36479
let
	to = TimerOutput()
	i = 4
	j = 2
	k = 3
	Ψ = rand(rng,Float32,i,j)
	α = rand(rng,Float32,i,k)

	# 	Ψ = ones(Float32,i,j)
	# α = ones(Float32,i,k)
	function obj(x)
		value = my_outer_product(Ψ,x)
	return sum(Array(value)[:])
	end
	backend = DifferentiationInterface.AutoZygote()
	@be DifferentiationInterface.gradient(x->obj(x),backend,α)
	A = DifferentiationInterface.gradient(x->obj(x),backend,α)
	display(A)

	backend = DifferentiationInterface.AutoForwardDiff()
	@be DifferentiationInterface.gradient(x->obj(x),backend,α)
	A = DifferentiationInterface.gradient(x->obj(x),backend,α)
	display(A)


	
end

# ╔═╡ 90971bd3-07ef-4239-b879-3260f0a9bb48
let
	to = TimerOutput()
	i = 4
	j = 2
	k = 3
	Ψ = rand(rng,Float32,i,j)
	α = rand(rng,Float32,i,k)

	# 	Ψ = ones(Float32,i,j)
	# α = ones(Float32,i,k)
	function obj(x)
		@tensor value[k, j] := Ψ[i, k] * x[i, j]
	return sum(Array(value)[:])
	end
	backend = DifferentiationInterface.AutoZygote()
	@be DifferentiationInterface.gradient(x->obj(x),backend,α)
	A = DifferentiationInterface.gradient(x->obj(x),backend,α)
	display(A)

	cus_dev = zeros(i,k)
	for ii in 1:i
		for kk in 1:k
			for jj in 1:j
				cus_dev[ii,kk] +=Ψ[ii,jj]
			end
		end
	end
	display(cus_dev)
	# display(α)
end

# ╔═╡ ca2d043c-5045-49e2-9c2a-736e8fbf8742
# let
# 	to = TimerOutput()
# 	i = 100
# 	j = 8
# 	k = 3
# 	Ψ = cu(rand(rng,Float32,i,j))
# 	α = cu(rand(rng,Float32,i,k))
# 	function obj(x)
# 		@ein value[k, j] := Ψ[i, k] * x[i, j]
# 	return mean(Array(value)[:])
# 	end
# 	backend = DifferentiationInterface.AutoEnzyme()
# 	@be DifferentiationInterface.gradient(x->obj(x),backend,α)
# 	DifferentiationInterface.gradient(x->obj(x),backend,α)
# end

# ╔═╡ c8dfa598-cc5c-43d0-a448-d1421264f8a0
let
	to = TimerOutput()
	i = 100
	j = 8
	k = 3
	Ψ = cu(rand(rng,Float32,i,j))
	α = cu(rand(rng,Float32,i,k))
	function obj(x)
		@ein value[k, j] := Ψ[i, k] * x[i, j]
	return mean(Array(value)[:])
	end
	backend = DifferentiationInterface.AutoZygote()
	@be DifferentiationInterface.gradient(x->obj(x),backend,α)
	DifferentiationInterface.gradient(x->obj(x),backend,α)
end

# ╔═╡ 1496e7a9-4230-4fcb-9cb4-40dbba7addf0
# let
# 	to = TimerOutput()
# 	i = 100
# 	j = 8
# 	k = 3
# 	Ψ = cu(rand(rng,Float32,i,j))
# 	α = cu(rand(rng,Float32,i,k))
# 	function obj(x)
# 		out = outer_product_kernel(Ψ,x)
# 	return mean(Array(out)[:])
# 	end
# 	backend = DifferentiationInterface.AutoZygote()
# 	DifferentiationInterface.gradient(x->obj(x),backend,α)
# end

# ╔═╡ 5d5efb70-2cd6-47ea-b374-ba47cf71fce1
# let
# 	to = TimerOutput()
# 	i = 100
# 	j = 8
# 	k = 3
# 	Ψ = cu(rand(rng,Float32,i,j))
# 	α = cu(rand(rng,Float32,i,k))
# 	function obj(x)
# 		out = outer_product_kernel(Ψ,x)
# 	return mean(Array(out)[:])
# 	end
# 	backend = DifferentiationInterface.AutoEnzyme()
# 	DifferentiationInterface.gradient(x->obj(x),backend,α)
# end

# ╔═╡ b71dc7b0-405b-4671-a9a3-bf066f50d792
# let
# 	to = TimerOutput()
# 	i = 100
# 	j = 8
# 	k = 3
# 	Ψ = cu(rand(rng,Float32,i,j))
# 	α = cu(rand(rng,Float32,i,k))
# 	backend = DifferentiationInterface.AutoForwardDiff()
# 	DifferentiationInterface.gradient(x->obj(Ψ,x),backend,α)
# end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Bumper = "8ce10254-0962-460f-a3d8-1f77fea1446e"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
Chairmarks = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
DispatchDoctor = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
Einsum = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TensorOperations = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
TimerOutputs = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
Tullio = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[compat]
Bumper = "~0.6.0"
CUDA = "~5.4.3"
ChainRulesCore = "~1.24.0"
Chairmarks = "~1.2.1"
DifferentiationInterface = "~0.5.8"
DispatchDoctor = "~0.4.10"
Einsum = "~0.4.1"
Enzyme = "~0.12.21"
ForwardDiff = "~0.10.36"
KernelAbstractions = "~0.9.22"
OMEinsum = "~0.8.2"
ReverseDiff = "~1.15.3"
Statistics = "~1.11.1"
TensorOperations = "~5.0.0"
TimerOutputs = "~0.5.24"
Tullio = "~0.3.7"
Zygote = "~0.6.70"
cuTENSOR = "~2.1.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.0-rc1"
manifest_format = "2.0"
project_hash = "3b54addef2884f4703bc0f8ec5ac2e41e090c9d8"

[[deps.ADTypes]]
git-tree-sha1 = "1f3835083f5b40fc01a3c87e64bc3275cf447481"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.5.4"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5c9b74c973181571deb6442d41e5c902e6b9f38e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.12.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "2c7cc21e8678eff479978a0a2ef5ce2f51b63dff"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BatchedRoutines]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "441db9f0399bcfb4eeb8b891a6b03f7acc5dc731"
uuid = "a9ab73d0-e05c-5df1-8fde-d6a4645b8d8e"
version = "0.2.2"

[[deps.BetterExp]]
git-tree-sha1 = "dd3448f3d5b2664db7eceeec5f744535ce6e759b"
uuid = "7cffe744-45fd-4178-b173-cf893948b8b7"
version = "0.1.0"

[[deps.Bumper]]
deps = ["StrideArraysCore"]
git-tree-sha1 = "aa2fc4ee0754a4ec23208961d4d40f154157f5a3"
uuid = "8ce10254-0962-460f-a3d8-1f77fea1446e"
version = "0.6.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
git-tree-sha1 = "fdd9dfb67dfefd548f51000cc400bb51003de247"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.4.3"
weakdeps = ["ChainRulesCore", "EnzymeCore", "SpecialFunctions"]

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "97df9d4d6be8ac6270cb8fd3b8fc413690820cbd"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.9.1+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f3b237289a5a77c759b2dd5d4c2ff641d67c4030"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.3.4"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "afea94249b821dc754a8ca6695d3daed851e1f5a"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.14.1+0"

[[deps.CUTENSOR_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2ad02c8180d94cca10336fc5646a7e24ab4aa268"
uuid = "35b6c64b-1ee1-5834-92a3-3f624899209a"
version = "2.0.1+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "227985d885b4dbce5e18a96f9326ea1e836e5a03"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.69.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Chairmarks]]
deps = ["Printf"]
git-tree-sha1 = "989bd3bb757ac0231fc77103e1b516e05c7d21f1"
uuid = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
version = "1.2.1"
weakdeps = ["Statistics"]

    [deps.Chairmarks.extensions]
    StatisticsChairmarksExt = ["Statistics"]

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "Compat", "DocStringExtensions", "FillArrays", "LinearAlgebra", "PackageExtensionCompat", "SparseArrays", "SparseMatrixColorings"]
git-tree-sha1 = "b04acb57861ae60ce548e6041795cac9e4ad96fa"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.5.8"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = "Enzyme"
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = "ForwardDiff"
    DifferentiationInterfacePolyesterForwardDiffExt = "PolyesterForwardDiff"
    DifferentiationInterfaceReverseDiffExt = "ReverseDiff"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTapirExt = "Tapir"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tapir = "07d77754-e150-4737-8c94-cd238a1fb45b"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "32d236e685d028f5bc808aae0634b58aac5128f0"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.10"
weakdeps = ["ChainRulesCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "Preferences", "Printf", "Random"]
git-tree-sha1 = "cd26bc551756abf6020535613d0b6056962d67b9"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.12.21"
weakdeps = ["ChainRulesCore", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "d445df66dd8761a4c27df950db89c6a3a0629fe7"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.7.7"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "3d38f0669f5ce1206f6bdcb6876ee01caf3cbd65"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.128+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "04661708f5301394a1f1be86a07a89e835900db6"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "10.2.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "ab29216184312f99ff957b32cd63c2fe9c928b91"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.26.7"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "950c3717af761bc3ff906c2e8e52bd83390b6ec2"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.14"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "d0448cebd5919e06ca5edc7a264631790de810ec"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.22"
weakdeps = ["EnzymeCore"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "020abd49586480c1be84f57da0017b5d3db73f7c"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "8.0.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c2636c264861edc6d305e6b4d528f09566d24c5e"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.30+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LRUCache]]
git-tree-sha1 = "b3cc6698599b10e652832c2f23db3cab99d51b59"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.1"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "53046f0483375e3ed78e49190f1154fa0a4083a1"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "0.3.4"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ce3269ed42816bf18d500c9f63418d4b0d9f5a3b"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.1.0+2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OMEinsum]]
deps = ["AbstractTrees", "BatchedRoutines", "ChainRulesCore", "Combinatorics", "LinearAlgebra", "MacroTools", "OMEinsumContractionOrders", "Test", "TupleTools"]
git-tree-sha1 = "fd0ce51747b27676ecb5cf21ea652f20e1a28c70"
uuid = "ebe7aa44-baf0-506c-a96f-8464559b3922"
version = "0.8.2"

    [deps.OMEinsum.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"

    [deps.OMEinsum.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.OMEinsumContractionOrders]]
deps = ["AbstractTrees", "BetterExp", "JSON", "SparseArrays", "Suppressor"]
git-tree-sha1 = "b0cba9f4a6f021a63b066f0bb29a6fd63c93be44"
uuid = "6f22d1fd-8eed-4bb7-9776-e7d684900715"
version = "0.8.3"

    [deps.OMEinsumContractionOrders.extensions]
    KaHyParExt = ["KaHyPar"]

    [deps.OMEinsumContractionOrders.weakdeps]
    KaHyPar = "2a6221f6-aa48-11e9-3542-2d9e0ef01880"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "195e0a19842f678dd3473ceafbe9d82dfacc583c"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.4.1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "66b20dd35966a748321d3b2537c4584cf40387c7"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "cc6cd622481ea366bb9067859446a8b01d92b468"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.15.3"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ff11acffdb082493657550959d4feb4b6149e73a"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.5"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "Compat", "DocStringExtensions", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "eed2446b3c3dd58f6ded3168998b8b2cb3fc9229"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.3.3"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d2fdac9ff3906e27f7a618d47b676941baa6c80c"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.10"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "8963e5a083c837531298fc41599182a759a87a6d"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.1"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "f35f6ab602df8413a50c4a25ca14de821e8605fb"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.7"

[[deps.Strided]]
deps = ["LinearAlgebra", "StridedViews", "TupleTools"]
git-tree-sha1 = "bd9bd1c70cfc115cc3a30213fc725125a6b43652"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.1.0"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "2917996ce0fa6b8a3a85240a5e9ff930e2aeaa43"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.3.1"
weakdeps = ["CUDA"]

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructIO]]
deps = ["Test"]
git-tree-sha1 = "010dc73c7146869c042b49adcdb6bf528c12e859"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.6.0+0"

[[deps.Suppressor]]
deps = ["Logging"]
git-tree-sha1 = "9143c41bd539a8885c79728b9dedb0ce47dc9819"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.7"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorOperations]]
deps = ["LRUCache", "LinearAlgebra", "PackageExtensionCompat", "PtrArrays", "Strided", "StridedViews", "TupleTools", "VectorInterface"]
git-tree-sha1 = "619e4a9fb0c216081a6483b0bf02261dab409828"
uuid = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
version = "5.0.0"
weakdeps = ["Bumper", "CUDA", "ChainRulesCore", "cuTENSOR"]

    [deps.TensorOperations.extensions]
    TensorOperationsBumperExt = "Bumper"
    TensorOperationsChainRulesCoreExt = "ChainRulesCore"
    TensorOperationscuTENSORExt = ["cuTENSOR", "CUDA"]

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "5a13ae8a41237cff5ecf34f73eb1b8f42fff6531"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.24"

[[deps.Tullio]]
deps = ["DiffRules", "LinearAlgebra", "Requires"]
git-tree-sha1 = "6d476962ba4e435d7f4101a403b1d3d72afe72f3"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
version = "0.3.7"

    [deps.Tullio.extensions]
    TullioCUDAExt = "CUDA"
    TullioChainRulesCoreExt = "ChainRulesCore"
    TullioFillArraysExt = "FillArrays"
    TullioTrackerExt = "Tracker"

    [deps.Tullio.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.TupleTools]]
git-tree-sha1 = "41d61b1c545b06279871ef1a4b5fcb2cac2191cd"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "bf2c553f25e954a9b38c9c0593a59bb13113f9e5"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.5"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "7aff7d62bffad9bba9928eb6ab55226b32a351eb"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.4.6"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "19c586905e78a26f7e4e97f81716057bd6b1bc54"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.70"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.cuTENSOR]]
deps = ["CEnum", "CUDA", "CUDA_Runtime_Discovery", "CUTENSOR_jll", "LinearAlgebra", "Printf"]
git-tree-sha1 = "e2c4b570ce57694054f88379cd3bcf98f7a06a0d"
uuid = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"
version = "2.1.1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═6ddcc330-436f-11ef-28f8-b76e24aed2b4
# ╠═277bf495-88e0-4970-ae3f-0e9dd488beff
# ╠═c95eb473-bced-4fa4-a686-c07a28992e48
# ╠═af1a1731-3e1b-4ebe-a527-3103a5bda87a
# ╠═623cd8e8-bf2e-4a96-95ff-2ce5ee928737
# ╠═cdfcc712-c291-4db4-acde-fe0d14f8cb25
# ╠═b12a9835-a1fb-4b20-8992-85a45f347d5f
# ╠═2ba00792-7be0-4dd1-ba76-cec8796e6d47
# ╠═210a3ec9-2c45-4520-9cd2-674a3d6f9fad
# ╠═b23aa282-2c59-43ae-96f9-61a268388747
# ╠═5aeac385-f1a1-4a0f-8f50-5f1fc8fdddbd
# ╠═4abbcdff-2d44-4a93-beba-6bc89ca20454
# ╠═2b49ab93-593c-4633-8548-426d6de50efb
# ╠═5ec804ae-9e8e-4cae-9f71-cb9347405d5b
# ╠═a472ad1f-122e-4b5c-990a-23330721e21f
# ╠═69554245-386a-4c9d-9e48-53115b58f34f
# ╠═ced5c143-1227-49c8-9cea-3ad680745cb6
# ╠═72cef223-ef79-4f21-b5c0-d3d3eb097ace
# ╠═638b7fa7-be3b-4c41-abf6-19d26035d47a
# ╠═08d56edc-827a-4429-8e64-7eee13c36479
# ╠═90971bd3-07ef-4239-b879-3260f0a9bb48
# ╠═ca2d043c-5045-49e2-9c2a-736e8fbf8742
# ╠═c8dfa598-cc5c-43d0-a448-d1421264f8a0
# ╠═1496e7a9-4230-4fcb-9cb4-40dbba7addf0
# ╠═5d5efb70-2cd6-47ea-b374-ba47cf71fce1
# ╠═b71dc7b0-405b-4671-a9a3-bf066f50d792
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
