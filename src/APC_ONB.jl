
function mean(x)
	s = zero(eltype(x))
	for i in x
		s += i
	end
	return s / length(x)
end

function var(x::AbstractArray)
	m = mean(x)
	s = zero(eltype(x))
	for i in x
		s += (i - m)^2
	end
	return s / (length(x) - 1)
end

function aPC_OrthonormalBasis(Data, Degree)
	d = Degree #Degree of polinomial expansion
	dd = d + 1 #Degree of polinomial for roots defenition
	L_norm = 1 # L-norm for polnomial normalization
	NumberOfDataPoints = length(Data)
	# @info "Construction of Arbitrary Polynomial Basis" --> We do d+2 x d+2, to have one order higher polynomials to get the gaussian quadrature poitns in the end!!!

	MeanOfData = mean(Data)
	VarOfData = var(Data)
	Data = Data ./ MeanOfData
	m = zeros(2 * dd + 2)
	for i ∈ 0:(2*dd+1)
		m[i+1] = sum(Data .^ i) / NumberOfDataPoints # Raw Moments
	end
	poly = zeros(dd + 1, dd + 1)
	MHankel = zeros(dd + 1, dd + 1) # Allocate once for all :)
	for degree ∈ 0:dd
		Hankel = @views MHankel[1:degree+1, 1:degree+1]
		Hankel .= 0.0
		fr = copy(Hankel)
		# fr1 = copy(fr)
		Vc = zeros(degree + 1)
		PolyCoeff_NonNorm = copy(Hankel)

		for i ∈ 0:degree
			for j ∈ 0:degree
				if i < degree
					Hankel[i+1, j+1] = m[i+j+1] # put in the moment
				elseif (i == degree) && (j < degree)
					Hankel[i+1, j+1] = 0.0
				elseif (i == degree) && (j == degree)
					Hankel[i+1, j+1] = 1.0
				end
			end
			# fr1 = copy(Hankel); # Control Hankel only considering the raw moments without division by max(abs) in each row
			Hankel[i+1, :] = Hankel[i+1, :] / maximum(abs.(Hankel[i+1, :]))
		end

		for i ∈ 0:degree
			if (i < degree)
				Vc[i+1] = 0
			elseif (i == degree)
				Vc[i+1] = 1
			end
		end

		# inv_Mm = pinv(Hankel, atol=1e-6)
		# display(Hankel)
		# dt = copy(Vc)
		# fr .= Hankel 
		Vp = Hankel \ Vc
		PolyCoeff_NonNorm[degree+1, 1:degree+1] .= Vp
		if (100 * abs(sum(abs.(Hankel * PolyCoeff_NonNorm[degree+1, 1:degree+1])) - sum(abs.(Vc))) > 0.5)
			@warn "Computational error of the linear solver is too high"
		end


		#Normalization of polynomial coefficients
		P_norm = 0
		for i ∈ 1:NumberOfDataPoints
			Poly = 0
			for k ∈ 0:degree
				Poly += PolyCoeff_NonNorm[degree+1, k+1] * Data[i]^k
			end
			P_norm += Poly^2 / NumberOfDataPoints
		end

		for k ∈ 0:degree
			poly[degree+1, k+1] = PolyCoeff_NonNorm[degree+1, k+1] / sqrt(P_norm)
		end


	end

	# Backward linear transformation to the data space
	Data = Data * MeanOfData
	for k ∈ 1:lastindex(poly, 2)
		poly[:, k] = poly[:, k] ./ (MeanOfData^(k - 1))
	end

	#%% Data-driven Arbitrary Orthonormal Polynomial Basis
	OrthonormalBasis = poly
	return OrthonormalBasis # SMatrix{dd+1,dd+1}(
end
