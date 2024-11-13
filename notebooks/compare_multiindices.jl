### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ dd009330-66de-11ef-0d13-b5d4100be938
begin
    using Chairmarks
    using LinearAlgebra
    using Test
    using DispatchDoctor
    using LazyArrays
    using PrettyChairmarks
    using FastBroadcast
end

# ╔═╡ 5e17ee6e-9feb-4f29-811b-01ff4362520e


# ╔═╡ 312edb2b-4b0d-4390-b22b-29af90394598
let
    d = 3
    n = 5
    aPCE_MultivariatePolynomialDegrees(d, n)
end

# ╔═╡ 0b6d8b47-c4b6-4d67-94b6-ad0f2347c8c4
let
    d = 3
    n = 5
    calculateMultiIndices(d, n)
end

# ╔═╡ 01558ceb-89c0-4fd0-aae1-f0f72d44243c
let
    d = 3
    n = 5
    @bs aPCE_MultivariatePolynomialDegrees(d, n) seconds = 2
end

# ╔═╡ ebd64888-eff2-4dc8-acf8-79093890f5e6
let
    d = 3
    n = 5
    @bs calculateMultiIndices(d, n) seconds = 2
end

# ╔═╡ a6574509-35d1-46dd-b963-510eed3c56c6
let
    d = 7
    n = 10
    @bs numberPolynomials(d, n) seconds = 5
end

# ╔═╡ ba2426ae-c6bd-42ea-8ebf-fb1f1120080a
let

    d = 7
    n = 10
    PrettyChairmarks.@bs mynumberPolynomials(d, n) seconds = 2
end

# ╔═╡ b3577f44-462a-4595-91f6-d0c6a900da01
let
    d = 3
    n = 10
    Ps = calculateMultiIndices(d, n)
    @show findUnivariateIndices(1, Ps)
    @bs findUnivariateIndices(1, Ps)  seconds = 2
end

# ╔═╡ adb7f181-03f6-4c94-98cb-7c4ab112fd39
@stable function mynumberPolynomials(n, d)
    x, y = max(d, n), min(d, n)
    return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end

# ╔═╡ c60fff0d-07bb-4569-a568-ae150e0226d1
begin

    @stable function calculateMultiIndices(d::Int, n::Int)
        # d denotes dimension of random variables/number of sources of uncertainty,
        # n the maximum degree of multivariate basis
        # function to calculate indices of multivariate basis following the algorithm
        # from "Spectral Methods for Uncertainty Quantiﬁcation; Le Maitre, Knio;2014" p.516-517
        # No::BigInt = factorial(BigInt(d+n))/(factorial(Bigint(d))*factorial(BigInt(n)));  #number of polynomials of multivariate basis
        n < 0 && throw(DomainError(n, "maximum degree must be non-negative"))
        d <= 0 && throw(DomainError(d, "number of uncertainties must be positive"))
        # catch case n == 0 --> No-d==0
        n == 0 && return zeros(Int64, 1, d)
        # non-pathological cases begin here
        No = numberPolynomials(d, n)
        inds = vcat(zeros(Int64, 1, d), Matrix(1I, d, d), zeros(Int64, No - d - 1, d))  #initiate index matrix for basis
        pi = ones(Int64, No, d)

        for k in 2:No
            g = 0
            for l in 1:d
                pi[k, l] = sum(pi[k - 1, :]) - g
                g = g + pi[k - 1, l]
            end
        end

        P = d + 1
        for k in 2:n
            L = P
            for j in 1:d, m in (L - pi[k, j] + 1):L
                P += 1
                inds[P, :] = inds[m, :]
                inds[P, j] = inds[P, j] + 1
            end
        end

        return inds
    end

    """
        computes the number of polynomials with a multivariate basis
        `(d+n)!/(d!+n!)`
    """

    @stable function numberPolynomials(d::Int64, n::Int64)
        x, y = max(d, n), min(d, n)
        return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y)))
    end

    """
        findUnivariateIndices(i::Int,ind::AbstractMatrix{Int64,2})
    
    Given the multi-index `ind` this function returns all entries of the multivariate basis
    that correspond to the `i`th univariate basis.
    """
    @stable function findUnivariateIndices(i::Int, ind::AbstractMatrix{Int})
        l, p = size(ind)
        i > p && throw(DomainError((i, p), "basis is $p-variate, you requested $i-variate"))
        deg = ind[end, end]
        deg < 0 && throw(DomainError(deg, "invalid degree"))
        col = ind[:, i]
        myind = zeros(Int64, deg)
        for deg_ in 1:deg
            myind[deg_] = findfirst(x -> x == deg_, col)
        end
        pushfirst!(myind, 1)
    end
end

# ╔═╡ 3e465f39-460d-4b5e-a937-a5a9547b572d
begin

    # Sort mean and variance at same time
    struct SpecialCoSorterElement{T1, T2, T3}
        x::T1
        z::T2
        y::T3
    end

    struct CoSorter{T1, T2, T3, A <: AbstractVecOrMat{T1}, B <: AbstractVecOrMat{T2}, C <: AbstractVecOrMat{T3}} <: AbstractVector{SpecialCoSorterElement{T1, T2, T3}}
        sortarray::A
        otherarray::B
        coarray::C
    end

    Base.size(c::CoSorter) = size(c.sortarray)
    Base.getindex(c::CoSorter, i...) =
        SpecialCoSorterElement(getindex(c.sortarray, i...), getindex(c.otherarray, i...), getindex(c.coarray, i...))
    Base.setindex!(c::CoSorter, t::SpecialCoSorterElement, i...) =
        (setindex!(c.sortarray, t.x, i...); setindex!(c.coarray, t.y, i...); c)

    Base.isless(a::SpecialCoSorterElement, b::SpecialCoSorterElement) = isless(a.x, b.x) || (a.x == b.x && isless(a.z, b.z))


    Base.Sort.defalg(v::C) where {T <: Union{Number, Missing}, C <: CoSorter{T}} =
        Base.DEFAULT_UNSTABLE
    @stable function special_sort_two_arrays!(x::AbstractArray, y::AbstractArray)
        T = CoSorter(x[:, 1], x[:, 2], y)
        sort!(T)
        x = T.sortarray
        y = T.coarray
    end


    @stable function aPCE_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T, s_marginals::F, s_interactions::F) where {T <: Integer, F <: Real}
        # Initialize the indices for the first parameter
        @stable function get_stats(r)::Array{F} # Returns sum, nzeros, mean, var, min,max
            o = Series(Mean(), Variance(), Extrema())
            n = length(r)
            summe = 0
            n_zeros = 0
            @inbounds for e in 1:n
                if Base.iszero(r[e])
                    n_zeros += 1
                else
                    summe += r[e]
                end
                fit!(o, r[e])
            end
            meanval = summe / n
            meanval, varval, mm = value(o)
            return [summe, n_zeros, meanval, varval, mm.min, mm.max]
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


        indices = vcat(zeros(Int64, 1, d), Matrix(1I, d, d), zeros(Int64, No - d - 1, d))
        pi = ones(Int64, No, d)

        # Precompute sums to avoid recomputing inside the loop
        @inbounds for k in 2:No
            g = 0
            row_sum = sum(pi[k - 1, :])  # Precompute the sum of the previous row
            for l in 1:d
                pi[k, l] = row_sum - g
                g += pi[k - 1, l]  # Accumulate g in-place
            end
        end

        P = d + 1

        @inbounds for k in 2:n
            L = P
            for j in 1:d
                ms = L - pi[k, j] + 1
                me = L
                for m in ms:me
                    P += 1
                    @.. indices[P, :] = indices[m, :]
                    indices[P, j] += 1
                end
            end
        end


        if (s_marginals != 1.0) || (s_interactions != 1.0)
            stats = reduce(hcat, map(x -> get_stats(x), eachrow(indices)))'
            d_marginal_indices = T[]
            d_interactions_indices = T[]
            @inbounds for r in axes(stats, 1) # Go over columns
                if stats[r, 1] <= max_degree
                    if stats[r, 2] == (num_dimensions - 1)
                        push!(d_marginal_indices, r)
                    elseif (num_dimensions - stats[r, 2]) >= 1
                        push!(d_interactions_indices, r)
                    end
                end
            end
            sorting_d_marginal = stats[d_marginal_indices, 3:4]
            sorting_d_interactions = stats[d_interactions_indices, 3:4]
            all_marginals = 1:length(d_marginal_indices) |> collect
            all_interactions = 1:length(d_interactions_indices) |> collect
            if length(all_marginals) > 1
                special_sort_two_arrays!(sorting_d_marginal, all_marginals)
            end
            if length(all_interactions) > 1
                special_sort_two_arrays!(sorting_d_interactions, all_interactions)
            end
            keeper_marginals = d_marginal_indices[filter_by_percentage(all_marginals, s_marginals)]
            keeper_interactions = d_interactions_indices[filter_by_percentage(all_interactions, s_interactions)]
            idxkeep = vcat(keeper_marginals, keeper_interactions)
            indices = @views indices[idxkeep, :]
            indices = vcat(indices, Base.zeros(T, num_dimensions)')
            indices = sortslices(hcat(vec(sum(indices; dims = 2)), indices); dims = 1, rev = false)[:, 2:end]
        end

        return indices::Matrix{T} # from sparse to matrix.
    end

end

# ╔═╡ f71c4417-db30-46e2-a823-88d85abf221d
md"""

---

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chairmarks = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
DispatchDoctor = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PrettyChairmarks = "aafa11c5-44f9-44a1-b829-427e6ce1ffc2"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
Chairmarks = "~1.2.1"
DispatchDoctor = "~0.4.14"
FastBroadcast = "~0.3.5"
LazyArrays = "~2.2.0"
PrettyChairmarks = "~1.0.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.0-rc3"
manifest_format = "2.0"
project_hash = "e1d3a5545005d8f4db053bd6ac3b6c2d55a1f80d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "f54c23a5d304fb87110de62bace7777d59088c34"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.15.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
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
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "0dd7edaff278e346eb0ca07a7e75c9438408a3ce"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.10.3"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

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

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "c2acd1de2c4c357928f9fb6b60b402d914621378"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.14"

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

    [deps.DispatchDoctor.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "ab1b34570bcdf272899062e1a56285a53ecaae08"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.3.5"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fd0002c0b5362d7eb952450ad5eb742443340d6e"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.12.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "507b423197fdd9e77b74aa2532c0a05eb7eb4004"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.2.0"

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

    [deps.LazyArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

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

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "6d38fea02d983051776a856b7df75b30cf9a3c1f"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.16"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

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

[[deps.PrettyChairmarks]]
deps = ["Chairmarks", "Printf", "Statistics"]
git-tree-sha1 = "6ced454f88c64b178f7222de91ebcb253c7a4f0f"
uuid = "aafa11c5-44f9-44a1-b829-427e6ce1ffc2"
version = "1.0.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╠═dd009330-66de-11ef-0d13-b5d4100be938
# ╠═5e17ee6e-9feb-4f29-811b-01ff4362520e
# ╠═312edb2b-4b0d-4390-b22b-29af90394598
# ╠═0b6d8b47-c4b6-4d67-94b6-ad0f2347c8c4
# ╠═01558ceb-89c0-4fd0-aae1-f0f72d44243c
# ╠═ebd64888-eff2-4dc8-acf8-79093890f5e6
# ╠═a6574509-35d1-46dd-b963-510eed3c56c6
# ╠═ba2426ae-c6bd-42ea-8ebf-fb1f1120080a
# ╠═b3577f44-462a-4595-91f6-d0c6a900da01
# ╟─f71c4417-db30-46e2-a823-88d85abf221d
# ╠═adb7f181-03f6-4c94-98cb-7c4ab112fd39
# ╠═c60fff0d-07bb-4569-a568-ae150e0226d1
# ╠═3e465f39-460d-4b5e-a937-a5a9547b572d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
