using LazyGrids
using Combinatorics
function aPC_MultivariatePolynomialDegrees_old(N, d)
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
	for i ∈ 1:1:size(UniqueDegreeCombinations, 1)
		DegreeWeight[i] = 0.0
		for j ∈ 1:1:N
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
    return indices
end

# function aPC_MultivariatePolynomialDegrees(N::Int64, d::Int64)
# 	# Input:
# 	# N: Number of uncertain parameters
# 	# d: Degree of polynomial expansion
# 	# Output:
# 	# PolynomialDegree: Multivariate Polynomial Degrees

# 	# Total number of terms:
# 	P = numberPolynomials(N, d)
# 	degrees = 0:d |> collect
# 	degreelist = copy(degrees)
# 	for idx ∈ 1:N-1
# 		degreelist = repeat(degreelist, N)
# 		first_column = repeat(degrees, length(degreelist) ÷ (d + 1) - 1)

# 	end


# 	terms |> display
# 	aPC_MultivariatePolynomialDegrees_old(N, d) |> display
# 	return terms
# end


function aPC_MultivariatePolynomialDegrees(num_dimensions::T, max_degree::T; use_p = false, p::Float64 = 0.85) where {T<:Integer}
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



function aPC_PsiPolynomialMatrix(apc::aPC{T}, TrainingInput) where {T <: Real}
	NumberOfTerms, InputDimensions = size(apc.MultivariatePolynomialDegrees)
	NCpoints = size(TrainingInput, 1)
	Psi = ones(T, NumberOfTerms, NCpoints)
	for i ∈ 1:NumberOfTerms  # For each term in the polynomial expansion
		for j ∈ 1:NCpoints  # For each input sample
			product = 1.0  # Initialize the product for this term and sample
			for ii ∈ 1:InputDimensions  # For each dimension of the input
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
