using LazyGrids

function aPC_MultivariatePolynomialDegrees(N, d)
	# Input:
	# N- Number of uncertain parameters
	# d - Degree of polynomial expansion
	# Output:
	# PolynomialDegree - Multivariate Polynomial Degrees 
	# Total number of terms
	P = numberPolynomials(N,d)
	# Possible Degrees
	UniqueDegreeCombinations = zeros((d+1)^N,d)
	PossibleDegrees = [collect(0:d) for _ in N:-1:1]
	if N==1
	    UniqueDegreeCombinations = PossibleDegrees[1]
	else 
		tmp = cat(collect(ndgrid_array(reverse(PossibleDegrees)...))...; dims=3)
		UniqueDegreeCombinations = reshape(tmp,:,N)
	end
	# Possible degree computation
	DegreeWeight=zeros(1,size(UniqueDegreeCombinations,1))
	for i=1:1:size(UniqueDegreeCombinations,1) 
	    DegreeWeight[i] = 0.0
	    for j=1:1:N
	        DegreeWeight[i] = DegreeWeight[i] + UniqueDegreeCombinations[i,j]
	    end
	end
	# Sorting of possible degree
	id = sortperm(DegreeWeight;dims=2)[:]
	SortDegreeCombinations = UniqueDegreeCombinations[id,:]
	# Multivariate Polynomial Degrees  
	reverse_columns!(SortDegreeCombinations)
	return SortDegreeCombinations[1:P,:]
end


function aPC_PsiPolynomialMatrix(apc::aPC{T}, TrainingInput) where {T<:Real}
    NumberOfTerms, InputDimensions = size(apc.MultivariatePolynomialDegrees)
    NCpoints = size(TrainingInput, 1)
    Psi = ones(T,  NumberOfTerms,NCpoints)
    for i = 1:NumberOfTerms  # For each term in the polynomial expansion
        for j = 1:NCpoints  # For each input sample
            product = 1.0  # Initialize the product for this term and sample
            for ii = 1:InputDimensions  # For each dimension of the input
                degree = apc.MultivariatePolynomialDegrees[i, ii] + 1  # Degree for this dimension, adjusted for 1-based indexing
                coeffs = apc.OrthonormalBasis[degree,1:degree, ii]  # Extract the coefficients for the polynomial
                p = Polynomials.Polynomial(coeffs)  # Create the polynomial
				x = reduce(hcat,TrainingInput)[ii,j]
				# @show degree coeffs p x p(x) 
				
                product *= p(x)  # Evaluate the polynomial at x and multiply
            end
            Psi[i,j] = product  # Assign the product to Psi matrix
        end
    end

    return Psi
end