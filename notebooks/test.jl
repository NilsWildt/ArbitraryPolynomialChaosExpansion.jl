### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ fba3984a-8d33-41db-821b-0dc7d1154c3f
begin
	using Test
	using LazyGrids
	using Combinatorics
	using PyCall
	using StructArrays
	using StaticArrays
	using BenchmarkTools
end

# ╔═╡ ea75b25d-d9d7-42c6-9473-e3cd9eb4234f
function lexsort(A::AbstractArray{T}; axis=1) where {T<:Number}
	sortslices(A;dims=1)
end


# ╔═╡ 13f21b0d-f2d0-49a0-aec5-2881a10b1441
begin
	A = Float64[0 1 3; 3 4 3; 2 2 3 ; 2 3 3]
	@info A
	@info lexsort(A)
end

# ╔═╡ f54c3ec4-87f3-4b86-a927-f7856012c319
py"""
import numpy as np
def sort_basis_indices(keys, graded=True, reverse=False):
    keys_ = np.atleast_2d(keys)    # convert to a 2D array
    if reverse:
        keys_ = keys_[::-1]
    # get indices from smallest to largest, giving the 1st row a higher importance
    indices = np.array(np.lexsort(keys_))
    if graded:
        indices = indices[np.argsort(
            np.sum(keys_[:, indices], axis=0))].T
    return indices


def get_polynomial_basis(max_degree, ndim):
    # Arrays with smallest order (0) to the max degree + 1
    start = np.zeros(ndim, dtype=int)
    stop = np.full(ndim, max_degree+1, dtype=int)   # Add +1 so np.arange(0, stop) fills up to degree "d"
    bound = stop.max()

    # To control the size of the arrays:
    dtype = np.uint8 if bound < 256 else np.uint16
    range_ = np.arange(bound, dtype=dtype)           # vector with values of "d" to consider
    # Initialize the indices for the first parameter (row-wise), based on the order range
    indices = range_[:, np.newaxis]  # list of orders, in order

    # Fill the combinatorics array, one dimension at a time:
    for idx in range(ndim - 1):

        # Repeats the current set of indices ndim times
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = np.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices) // bound)[:, np.newaxis]

        # Put the array "front" in front of the previous "indices" array, to do the combinations of dimensions <= idx
        indices = np.column_stack((front, indices))

        # Truncate at each step to keep memory usage low, dor idx > 0
        idx_to_keep = np.sum(indices, axis=-1) <= max_degree
        indices = indices[idx_to_keep]

    # Order in descending norm value (sum of all orders), giving priority to the first dimensions
    new_order = sort_basis_indices(keys=indices.T, reverse=False, graded=True)
    indices = indices[new_order]

    return indices
"""

# ╔═╡ bb7cb6ed-f56b-44ac-8a31-a5c6f22ad4b0
function reverse_columns!(x)
    for row in axes(x, 1)
        x[row, :] = reverse(x[row, :])
    end
end

# ╔═╡ 5199de26-b5b6-4d66-a37b-40efe98b3bcc
function numberPolynomials(n::Int64,d::Int64)
    x, y = max(d, n), min(d, n)
    return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |>Int
end

# ╔═╡ f4003dec-6382-49b8-8311-805f4b3816f0
function aPC_MultivariatePolynomialDegrees(num_dims,max_degree)
	# Input:
	# num_dims- Number of uncertain parameters
	# max_degree - Degree of polynomial expansion
	# Output:
	# PolynomialDegree - Multivariate Polynomial Degrees 
	# Total number of terms
	P = numberPolynomials(num_dims, max_degree)
	# Possible Degrees
	UniqueDegreeCombinations = zeros((max_degree + 1)^num_dims, max_degree)
	PossibleDegrees = [collect(0:max_degree) for _ in num_dims:-1:1]
	if num_dims == 1
		UniqueDegreeCombinations = PossibleDegrees[1]
	else
		tmp = cat(collect(ndgrid_array(reverse(PossibleDegrees)...))...; dims = 3)
		UniqueDegreeCombinations = reshape(tmp, :, num_dims)
	end
	# Possible degree computation
	DegreeWeight = zeros(1, size(UniqueDegreeCombinations, 1))
	for i ∈ 1:1:size(UniqueDegreeCombinations, 1)
		DegreeWeight[i] = 0.0
		for j ∈ 1:1:num_dims
			DegreeWeight[i] = DegreeWeight[i] + UniqueDegreeCombinations[i, j]
		end
	end
	# Sorting of possible degree
	id = sortperm(DegreeWeight; dims = 2)[:]
	SortDegreeCombinations = UniqueDegreeCombinations[id, :]
	# Multivariate Polynomial Degrees  
	# reverse_columns!(SortDegreeCombinations)
	return SortDegreeCombinations[1:P, :]
end

# ╔═╡ be87bfb0-ea5c-4658-ae52-962eaebd52af
function sort_basis_indices(keys; graded=false, reverse=false)
    if reverse
        reverse!(keys, dims=1)
    end
    indices = sortperm(keys[:,1])
    if graded
        sums = sum(keys[indices, :], dims=2)
        graded_indices = sortperm(sums,dims=1)
        indices = indices[graded_indices]
    end
    return keys[indices,:]
end

# ╔═╡ c9603520-d2a0-4ddb-8edc-18d94ed7d70e
function flip_columns!(arr)
    ncols = size(arr, 2)  # Number of columns
    for i in 1:(ncols ÷ 2)
        arr[:, [i, ncols - i + 1]] = arr[:, [ncols - i + 1, i]]
    end
    return arr
end

# ╔═╡ 29cbd1a5-c02f-4c87-ac4d-5c2877ba3dfb
function aPC_MultivariatePolynomialDegrees_new(num_dimensions::T, max_degree::T; use_p = false, p::Float64 = 0.85) where {T<:Integer}
	# Initialize the indices for the first parameter
	range_ = 0:max_degree |> collect 
	indices = reshape(range_, :, 1)  # Make it a column vector

	for di in 1:num_dimensions-1
		indices = repeat(indices, inner = (max_degree+1,1))
		front = repeat(range_, outer = div(lastindex(indices) ,(max_degree+1))÷di)
		indices = hcat(front,indices)
		if use_p
			# Apply truncation using p-norm sparsity
			idx_to_keep = vec(sum((indices ./ (num_dimensions + 1)) .^ p, dims = 2) .^ (1 / p) .<= 1)
			indices = indices[idx_to_keep, :]
		else
			indices = indices[vec(sum(indices; dims = 2)).<=max_degree, :]
		end
	end

	indices = hcat(vec(sum(indices; dims = 2)),indices)
	indices = sortslices(indices;dims=1,rev=false)[:,2:end]
 	reverse_columns!(indices)
	return indices
end

# ╔═╡ 1575cd30-f787-43b6-86ba-d8df4095ac63
let
	py"""
import numpy as np
def sort_basis_indices(keys, graded=True, reverse=False):
    keys_ = np.atleast_2d(keys)    # convert to a 2D array
    if reverse:
        keys_ = keys_[::-1]
    # get indices from smallest to largest, giving the 1st row a higher importance
    indices = np.array(np.lexsort(keys_))
    if graded:
        indices = indices[np.argsort(
            np.sum(keys_[:, indices], axis=0))].T
    return indices


def get_polynomial_basis(max_degree, ndim):
    # Arrays with smallest order (0) to the max degree + 1
    start = np.zeros(ndim, dtype=int)
    stop = np.full(ndim, max_degree+1, dtype=int)   # Add +1 so np.arange(0, stop) fills up to degree "d"
    bound = stop.max()

    # To control the size of the arrays:
    dtype = np.uint8 if bound < 256 else np.uint16
    range_ = np.arange(bound, dtype=dtype)           # vector with values of "d" to consider
    # Initialize the indices for the first parameter (row-wise), based on the order range
    indices = range_[:, np.newaxis]  # list of orders, in order

    # Fill the combinatorics array, one dimension at a time:
    for idx in range(ndim - 1):

        # Repeats the current set of indices ndim times
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = np.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices) // bound)[:, np.newaxis]

        # Put the array "front" in front of the previous "indices" array, to do the combinations of dimensions <= idx
        indices = np.column_stack((front, indices))

        # Truncate at each step to keep memory usage low, dor idx > 0
        idx_to_keep = np.sum(indices, axis=-1) <= max_degree
        indices = indices[idx_to_keep]

    # Order in descending norm value (sum of all orders), giving priority to the first dimensions
    new_order = sort_basis_indices(keys=indices.T, reverse=False, graded=True)
    indices = indices[new_order]

    return indices
"""
	
	max_degree = 3 # Maximum degree
	num_dims = 3

	@timev aPC_MultivariatePolynomialDegrees(num_dims,max_degree) 
	@timev aPC_MultivariatePolynomialDegrees_new(num_dims,max_degree) 
	@timev Array{Int64}(py"get_polynomial_basis($max_degree,$num_dims)")
end
	

# ╔═╡ 70469375-3ada-4f12-8cbb-862ac5ae3296
max_degree = 6 # Maximum degree

# ╔═╡ 4b4c92a4-139c-40e8-acec-e4ecc9bb2c31
num_dims = 4

# ╔═╡ d794f287-afd4-4c11-bef8-0016196653e5
@time aPC_MultivariatePolynomialDegrees_new(num_dims,max_degree) 

# ╔═╡ f76d6f0d-c523-46c0-bd4f-22f92544b6a0
@benchmark aPC_MultivariatePolynomialDegrees(num_dims,max_degree) 

# ╔═╡ 50425bfa-051a-49d1-b3c9-298e8786955d
@benchmark aPC_MultivariatePolynomialDegrees_new(num_dims,max_degree) 

# ╔═╡ e40b08f2-2681-423b-9691-bd0bc7ea5373
begin
	py"""
	import numpy as np
	def sort_basis_indices(keys, graded=True, reverse=False):
	    keys_ = np.atleast_2d(keys)    # convert to a 2D array
	    if reverse:
	        keys_ = keys_[::-1]
	    # get indices from smallest to largest, giving the 1st row a higher importance
	    indices = np.array(np.lexsort(keys_))
	    if graded:
	        indices = indices[np.argsort(
	            np.sum(keys_[:, indices], axis=0))].T
	    return indices
	
	
	def get_polynomial_basis(max_degree, ndim):
	    # Arrays with smallest order (0) to the max degree + 1
	    start = np.zeros(ndim, dtype=int)
	    stop = np.full(ndim, max_degree+1, dtype=int)   # Add +1 so np.arange(0, stop) fills up to degree "d"
	    bound = stop.max()
	
	    # To control the size of the arrays:
	    dtype = np.uint8 if bound < 256 else np.uint16
	    range_ = np.arange(bound, dtype=dtype)           # vector with values of "d" to consider
	    # Initialize the indices for the first parameter (row-wise), based on the order range
	    indices = range_[:, np.newaxis]  # list of orders, in order
	
	    # Fill the combinatorics array, one dimension at a time:
	    for idx in range(ndim - 1):
	
	        # Repeats the current set of indices ndim times
	        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
	        indices = np.tile(indices, (bound, 1))
	
	        # Stretches ranges over the new dimension.
	        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
	        front = range_.repeat(len(indices) // bound)[:, np.newaxis]
	
	        # Put the array "front" in front of the previous "indices" array, to do the combinations of dimensions <= idx
	        indices = np.column_stack((front, indices))
	
	        # Truncate at each step to keep memory usage low, dor idx > 0
	        idx_to_keep = np.sum(indices, axis=-1) <= max_degree
	        indices = indices[idx_to_keep]
	
	    # Order in descending norm value (sum of all orders), giving priority to the first dimensions
	    new_order = sort_basis_indices(keys=indices.T, reverse=False, graded=True)
	    indices = indices[new_order]
	
	    return indices
	"""
	@benchmark Array{Int64}(py"get_polynomial_basis($max_degree,$num_dims)")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
LazyGrids = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
BenchmarkTools = "~1.4.0"
Combinatorics = "~1.0.2"
LazyGrids = "~0.5.0"
PyCall = "~1.96.4"
StaticArrays = "~1.9.2"
StructArrays = "~0.6.17"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "d5350ebe4aa3295b5865be8e663c4602fbba091a"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "51cab8e982c5b598eea9c8ceaced4b58d9dd37c9"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LazyGrids]]
deps = ["Statistics"]
git-tree-sha1 = "f43d10fea7e448a60e92976bbd8bfbca7a6e5d09"
uuid = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
version = "0.5.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "7b0e9c14c624e435076d19aea1e5cbdec2b9ca37"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.2"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "1b0b1205a56dc288b71b1961d48e351520702e24"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.17"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

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

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"
"""

# ╔═╡ Cell order:
# ╠═fba3984a-8d33-41db-821b-0dc7d1154c3f
# ╠═13f21b0d-f2d0-49a0-aec5-2881a10b1441
# ╠═ea75b25d-d9d7-42c6-9473-e3cd9eb4234f
# ╠═f54c3ec4-87f3-4b86-a927-f7856012c319
# ╠═bb7cb6ed-f56b-44ac-8a31-a5c6f22ad4b0
# ╠═5199de26-b5b6-4d66-a37b-40efe98b3bcc
# ╠═f4003dec-6382-49b8-8311-805f4b3816f0
# ╠═be87bfb0-ea5c-4658-ae52-962eaebd52af
# ╠═c9603520-d2a0-4ddb-8edc-18d94ed7d70e
# ╠═29cbd1a5-c02f-4c87-ac4d-5c2877ba3dfb
# ╟─1575cd30-f787-43b6-86ba-d8df4095ac63
# ╠═70469375-3ada-4f12-8cbb-862ac5ae3296
# ╠═4b4c92a4-139c-40e8-acec-e4ecc9bb2c31
# ╠═d794f287-afd4-4c11-bef8-0016196653e5
# ╠═f76d6f0d-c523-46c0-bd4f-22f92544b6a0
# ╠═50425bfa-051a-49d1-b3c9-298e8786955d
# ╠═e40b08f2-2681-423b-9691-bd0bc7ea5373
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
