### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 8fb81c25-a9c5-4f13-8200-5d4eca597322
begin
	using DispatchDoctor
	using Statistics
	using ComponentArrays
	using Chairmarks
	using OnlineStats
end

# ╔═╡ 62ae54e2-0f12-45bf-af58-23322a0c4cbe


# ╔═╡ d6a97257-71be-4dd5-ac7a-ff5298620270
begin
	# Sort mean and variance at same time
	struct CoSorterElement{T1,T2,T3}
	    x::T1
		z::T2
	    y::T3
	end
	
	struct CoSorter{T1,T2,T3,A <: AbstractVecOrMat{T1},B <: AbstractVecOrMat{T2},C <: AbstractVecOrMat{T3}} <: AbstractVector{CoSorterElement{T1,T2,T3}}
	    sortarray::A
	    otherarray::B
	    coarray::C
	end
	
	Base.size(c::CoSorter) = size(c.sortarray)
	Base.getindex(c::CoSorter, i...) = 
	    CoSorterElement(getindex(c.sortarray, i...), getindex(c.otherarray, i...),getindex(c.coarray, i...))
	Base.setindex!(c::CoSorter, t::CoSorterElement, i...) = 
	    (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c) 

 Base.isless(a::CoSorterElement, b::CoSorterElement) =isless(a.x, b.x) || (a.x == b.x && isless(a.z, b.z))
		
    
	Base.Sort.defalg(v::C) where {T <: Union{Number,Missing},C <: CoSorter{T}} = 
	    Base.DEFAULT_UNSTABLE
	@stable function sort_two_arrays!(x::AbstractArray, y::AbstractArray)
	    T = CoSorter(x[:,1],x[:,2], y)
	    sort!(T)
	    x = T.sortarray
	    y = T.coarray
	end

	@stable function aPCE_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T, s_marginals::F, s_interactions::F)::Matrix{T} where {T<:Integer,F<:Real}
		# Initialize the indices for the first parameter
		@stable function get_stats(r) # Returns sum, nzeros, mean, var, min,max
			o = Series(Mean(),Variance(), Extrema())
			n = length(r)
			summe = 0
			n_zeros = 0
			@inbounds for e in 1:n
				if iszero(r[e])
					n_zeros += 1
				else
					summe += r[e]
				end
				fit!(o,r[e])
			end
			meanval = summe/n
			meanval,varval,mm = value(o)
			return [summe,n_zeros,meanval,varval,mm.min,mm.max]
		end	
		
	@stable function filter_by_percentage(array::AbstractArray, percentage)
	    # Ensure the percentage is within the valid range
	    if percentage < 0.0 || percentage > 1.0
	        throw(ArgumentError("Percentage must be between 0 and 1"))
	    end
	    n = length(array)
	    num_to_keep = round(Int, percentage * n)
	    return @views array[1:num_to_keep]
	end

	
		range_ = 0:max_degree |> collect
		indices = reshape(range_, :, 1)  # Make it a column vector
		@inbounds for di in 1:num_dimensions-1
			indices = repeat(indices, inner = (max_degree + 1, 1))
			front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di)
			indices = hcat(front,indices)
			indices = indices[vec(sum(indices; dims = 2)).<=max_degree, :]
		end

		stats = reduce(hcat,map(x->get_stats(x),eachrow(indices)))'
		d_marginal_indices = T[]
		d_interactions_indices = T[]
		@inbounds for r in axes(stats,1) # Go over columns
			if stats[r,1] <= max_degree
				if stats[r,2] == (num_dimensions - 1)
						push!(d_marginal_indices,r)
				elseif (num_dimensions - stats[r,2]) >= 1
							push!(d_interactions_indices,r)
					end
				end
		end

		sorting_d_marginal = stats[d_marginal_indices,3:4] 
		sorting_d_interactions = stats[d_interactions_indices,3:4]

		all_marginals= 1:length(d_marginal_indices) |> collect
		all_interactions = 1:length(d_interactions_indices)|> collect

		if length(all_marginals) > 1
			sort_two_arrays!(sorting_d_marginal,all_marginals)
		end
		if length(all_interactions) > 1
			sort_two_arrays!(sorting_d_interactions,all_interactions)
		end

		keeper_marginals = d_marginal_indices[filter_by_percentage(all_marginals,s_marginals)]
		keeper_interactions = d_interactions_indices[filter_by_percentage(all_interactions,s_interactions)]

		idxkeep = vcat(keeper_marginals,keeper_interactions)
		indices = @views indices[idxkeep,:]
		# indices = indices[[we_want_to_keep(i) for i in eachrow(indices)],:]
		# keep_mask = map(we_want_to_keep, eachrow(indices))
		# indices = indices[keep_mask, :]
		indices = vcat(indices,zeros(T,num_dimensions)')
		indices = sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
		# idx_to_keep = vec(sum((indices ./ (max_degree + 1)) .^ qnorm, dims = 2) .^ (1.0 / qnorm) .<= 1.0)
		# indices = indices[idx_to_keep, :]
# 			# @info "Q-norm removed $(length(idx_to_keep)-sum(idx_to_keep)) terms"
		# reverse_columns!(indices)
		# @info "The multivariate polynomial degrees:" indices
		# UnicodePlots.spy(sparse(indices)) |> display
		return indices
	end
end

# ╔═╡ edcd443a-40a0-4e78-98ef-1efe4ba1d7c6
let
	a = rand(4,2)
	a[1] = 0.0
	a[2] = 0.0
	
	a[1,2] = 0.5
	a[2,2] = 0.1
	
	display(a)
	b = 1:4 |> collect 
	sort_two_arrays!(a,b)
	@info b
end

# ╔═╡ 107808b0-8493-46f8-9a0f-d7e08915cf47
aPCE_MultivariatePolynomialDegrees(3,3,0.1,0.1)'

# ╔═╡ dbc69d40-280c-11ef-3f1b-8bf398dd82fe

@stable function aPCE_MultivariatePolynomialDegrees1(num_dimensions::T, max_degree::T, d_marginals::T, d_interactions::T)::Matrix{T} where {T<:Integer}
		# Initialize the indices for the first parameter


	
		@stable function we_want_to_keep(current_combination)::Bool
			scc = sum(current_combination) 
				if scc<= max_degree
					zeroinds = iszero.(current_combination)
					szeroinds = sum(zeroinds)
					if szeroinds == (num_dimensions - 1)
						if current_combination[.!zeroinds][1] <= d_marginals
							return true
						end
					elseif (num_dimensions - szeroinds) >= 1
						if scc <= d_interactions
							return true
						end
					end
				end
			return false
		end
		range_ = 0:max_degree |> collect
		indices = reshape(range_, :, 1)  # Make it a column vector
		@inbounds for di in 1:num_dimensions-1
			indices = repeat(indices, inner = (max_degree + 1, 1))
			front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di)
			indices = hcat(front,indices)
			indices = indices[vec(sum(indices; dims = 2)).<=max_degree, :]
		end
		# NOw kill the ones we don't want:
		# indices = indices[[we_want_to_keep(i) for i in eachrow(indices)],:]
		# keep_mask = map(we_want_to_keep, eachrow(indices))
		# indices = indices[keep_mask, :]
		# indices = vcat(indices,zeros(T,num_dimensions)')
		indices = sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
		# idx_to_keep = vec(sum((indices ./ (max_degree + 1)) .^ qnorm, dims = 2) .^ (1.0 / qnorm) .<= 1.0)
		# indices = indices[idx_to_keep, :]
# 			# @info "Q-norm removed $(length(idx_to_keep)-sum(idx_to_keep)) terms"
		# reverse_columns!(indices)
		# @info "The multivariate polynomial degrees:" indices
		# UnicodePlots.spy(sparse(indices)) |> display
		return indices
	end
	

# ╔═╡ 997de16e-295c-41a1-87e7-0ccc57122894
c = a[:,21]

# ╔═╡ af3a8d5f-8e63-4ad0-82e0-5d4dfa2482b6
d = [1,1,1,1]

# ╔═╡ 537ae22b-e1c3-4e28-b839-8324a6904549
begin
	@show get_stats(b)
	@b get_stats(b)
end

# ╔═╡ fbb75e80-6c3e-4e5b-b554-7a7fc4626060


# ╔═╡ 73cd051d-1f86-41c8-b296-e58f17be7f10
begin

end

# ╔═╡ a4fa955f-3cee-4371-9c43-f7a87e264c2d

@stable function aPCE_MultivariatePolynomialDegrees2(num_dimensions::T, max_degree, d_marginals, d_interactions)::Matrix{T} where {T<:Int64}
		# Initialize the indices for the first parameter
		@stable function filter_by_percentage(array::AbstractArray, percentage::Float64)
    # Ensure the percentage is within the valid range
    if percentage < 0.0 || percentage > 1.0
        throw(ArgumentError("Percentage must be between 0 and 1"))
    end
    n = length(array)
    num_to_keep = round(Int, percentage * n)
    return @views array[1:num_to_keep]
end

		@stable function we_want_to_keep(current_combination)::Bool
				if sum(current_combination) <= max_degree
					zeroinds = iszero.(current_combination)
					szeroinds = sum(zeroinds)
					if szeroinds == (num_dimensions - 1)
						if current_combination[.!zeroinds][1] <= d_marginals
							return true
						end
					elseif (num_dimensions - szeroinds) >= 1
						if sum(current_combination[.!zeroinds]) <= d_interactions
							return true
						end
					end
				end
			return false
		end
		range_ = 0:max_degree |> collect
		indices = reshape(range_, :, 1)  # Make it a column vector
		@inbounds for di in 1:num_dimensions-1
			indices = repeat(indices, inner = (max_degree + 1, 1))
			front = repeat(range_, outer = div(lastindex(indices), (max_degree + 1)) ÷ di)
			indices = hcat(front,indices)
			indices = indices[vec(sum(indices; dims = 2)).<=max_degree, :]
		end
		# NOw kill the ones we don't want:
		# indices = indices[[we_want_to_keep(i) for i in eachrow(indices)],:]
		keep_mask = map(we_want_to_keep, eachrow(indices))
		indices = indices[keep_mask, :]
		indices = vcat(indices,zeros(T,num_dimensions)')
		indices = sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
		# idx_to_keep = vec(sum((indices ./ (max_degree + 1)) .^ qnorm, dims = 2) .^ (1.0 / qnorm) .<= 1.0)
		# indices = indices[idx_to_keep, :]
# 			# @info "Q-norm removed $(length(idx_to_keep)-sum(idx_to_keep)) terms"
		# reverse_columns!(indices)
		# @info "The multivariate polynomial degrees:" indices
		# UnicodePlots.spy(sparse(indices)) |> display
		return indices
	end


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chairmarks = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
DispatchDoctor = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
OnlineStats = "a15396b6-48d5-5d58-9928-6d29437db91e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Chairmarks = "~1.2.1"
ComponentArrays = "~0.15.13"
DispatchDoctor = "~0.4.7"
OnlineStats = "~1.7.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "bb652bd9ad2a259513fa64ba1c2a6f56011861ed"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ed2ec3c9b483842ae59cd273834e5b46206d6dda"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.11.0"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

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

[[deps.ComponentArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "Functors", "LinearAlgebra", "PackageExtensionCompat", "StaticArrayInterface", "StaticArraysCore"]
git-tree-sha1 = "85d7d0c192e8eec909799737fe590f7d7ff0a6eb"
uuid = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
version = "0.15.13"

    [deps.ComponentArrays.extensions]
    ComponentArraysAdaptExt = "Adapt"
    ComponentArraysConstructionBaseExt = "ConstructionBase"
    ComponentArraysGPUArraysExt = "GPUArrays"
    ComponentArraysOptimisersExt = "Optimisers"
    ComponentArraysRecursiveArrayToolsExt = "RecursiveArrayTools"
    ComponentArraysReverseDiffExt = "ReverseDiff"
    ComponentArraysSciMLBaseExt = "SciMLBase"
    ComponentArraysTrackerExt = "Tracker"
    ComponentArraysTruncatedStacktracesExt = "TruncatedStacktraces"
    ComponentArraysZygoteExt = "Zygote"

    [deps.ComponentArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    TruncatedStacktraces = "781d530d-4396-4725-bb49-402e4bee1e77"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences", "TestItems"]
git-tree-sha1 = "ac16550f9edcecdff854d6514e8bcd718427922c"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.7"
weakdeps = ["ChainRulesCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8a66c07630d6428eaab3506a0eabfcf4a9edea05"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.11"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OnlineStats]]
deps = ["AbstractTrees", "Dates", "Distributions", "LinearAlgebra", "OnlineStatsBase", "OrderedCollections", "Random", "RecipesBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "29f0e2b369c22190f2500b4fe5af49052c3f0c3b"
uuid = "a15396b6-48d5-5d58-9928-6d29437db91e"
version = "1.7.0"

[[deps.OnlineStatsBase]]
deps = ["AbstractTrees", "Dates", "LinearAlgebra", "OrderedCollections", "Statistics", "StatsBase"]
git-tree-sha1 = "9a067a4ea67d1ebab4554b73792dd429f098387c"
uuid = "925886fa-5bf2-5e8e-b522-a9147a512338"
version = "1.7.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

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
git-tree-sha1 = "5d66818a39bb04bf328e92bc933ec5b4ee88e436"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.0"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TestItems]]
git-tree-sha1 = "8621ba2637b49748e2dc43ba3d84340be2938022"
uuid = "1c621080-faea-4a02-84b6-bbd5e436b8fe"
version = "0.1.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

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

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═8fb81c25-a9c5-4f13-8200-5d4eca597322
# ╠═62ae54e2-0f12-45bf-af58-23322a0c4cbe
# ╠═edcd443a-40a0-4e78-98ef-1efe4ba1d7c6
# ╠═d6a97257-71be-4dd5-ac7a-ff5298620270
# ╠═107808b0-8493-46f8-9a0f-d7e08915cf47
# ╠═dbc69d40-280c-11ef-3f1b-8bf398dd82fe
# ╠═997de16e-295c-41a1-87e7-0ccc57122894
# ╠═af3a8d5f-8e63-4ad0-82e0-5d4dfa2482b6
# ╠═537ae22b-e1c3-4e28-b839-8324a6904549
# ╠═fbb75e80-6c3e-4e5b-b554-7a7fc4626060
# ╠═73cd051d-1f86-41c8-b296-e58f17be7f10
# ╠═a4fa955f-3cee-4371-9c43-f7a87e264c2d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
