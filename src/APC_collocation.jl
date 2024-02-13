using Polynomials


function GaussianCollocation(apc::aPC{T};Strategy=:FT)::RowVecs{T} where {T<:Real}
 		AvailableCollocationPoints=zeros(ComplexF64,apc.input_dimensions,apc.ExpansionDegree+1)
        for d=1:apc.input_dimensions
            poly=apc.OrthonormalBasis[:,:,d]
			AvailableCollocationPoints[d,:] = Polynomials.roots(Polynomials.Polynomial(poly[apc.ExpansionDegree+2,:]))
			# AvailableCollocationPoints[d,:]  = AMRVW.roots(poly[apc.ExpansionDegree+2,:])
			
        end
	 AvailableCollocationPoints = reverse(AvailableCollocationPoints)
	
        PointsVector=1:apc.ExpansionDegree+1 |> collect
	 	UniqueCombinations = stack(reduce(vcat,collect(Iterators.product([PointsVector for i in 1:apc.input_dimensions]...))))' |> collect
	
        DigitalPointsWeight=zeros(size(UniqueCombinations,1))
	 
        for (i,r) in enumerate(eachrow(UniqueCombinations))
            DigitalPointsWeight[i] = sum(r)              
        end
	 
       
	 	index_SDPW = sortperm(reshape(DigitalPointsWeight,(:,1));dims=1)[:]
		 # display(index_SDPW)
        SortUniqueCombinations=UniqueCombinations[index_SDPW,:]

        TrainingInput = SortUniqueCombinations
      
    # @info "Generated Gaussian collocation input"
    return  RowVecs(Array{T}(view(Float64.(TrainingInput), :, reverse(1:size(TrainingInput,2)))))
end