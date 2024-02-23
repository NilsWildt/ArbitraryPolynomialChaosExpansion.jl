# Copyright 2024 wildt
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using LinearAlgebra
using RegularizedLeastSquares
using FastLevenbergMarquardt
using Zygote
# using ForwardDiff
using NonlinearSolve, StaticArrays
using IterativeSolvers
using Enzyme
using Hyperopt
using Preconditioners
using CUDA
import Optim: NewtonTrustRegion, Options, optimize, minimizer, minimum, LBFGS
import RegularizationTools: Lₖx₀, solve, RegularizationProblem, setupRegularizationProblem, to_general_form, to_standard_form, gcv_tr, gcv_svd, invert, Lₖ, NelderMead


function numberPolynomials(n::Int64, d::Int64)
	x, y = max(d, n), min(d, n)
	return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |> Int
end


function reverse_columns!(x)
	for row in axes(x, 1)
		x[row, :] = reverse(x[row, :])
	end
end

function train!(apc::aPC{T}, TrainingInput::RowVecs, TrainingOutput::RowVecs) where {T <: Real}
	@info "=> aPC Toolbox: Training Arbitrary Polynomial Chaos ..."
	Psi = aPC_PsiPolynomialMatrix(apc, TrainingInput)'
	to = reduce(vcat, TrainingOutput)


	# reg = L1Regularization(1e-9)
	# S = createLinearSolver(ADMM, Psi; reg=reg, iterations=200,verbose=true)
	#  out =  solve!(S, to)
	#  apc.ExpansionCoefficients .= out




	# reg = RegularizedLeastSquares.L1Regularization(1e-1)
	# pre = CholeskyPreconditioner(Psi'*Psi,2)
	# solver = createLinearSolver(ADMM, Psi; reg=reg, iterations=8000,verbose=false,rho=0.1,precon =pre,normalizeReg =SystemMatrixBasedNormalization())  # 
	# apc.ExpansionCoefficients  .= RegularizedLeastSquares.solve!(solver,to)


	# u0 = apc.ExpansionCoefficients 
	## See p. 96 Sullivan UQ 
	# Also check https://gregorygundersen.com/blog/2019/09/12/practical-gp-regression/

	# function ℓπ_old(A, y, λ₁::T = 1e-12, λ₂::T = 0.1) where {T <: Real}
	# 	Q = (I * λ₁)
	# 	Qi = inv(Q)
	# 	R = I * λ₂
	# 	Ri = inv(R)
	# 	AQiA = A' * Qi * A
	# 	P = AQiA + Ri
	# 	Ki = pinv(AQiA) * A' * Qi
	# 	ubar = zeros(size(A, 2))
	# 	u = inv(P) * (AQiA * Ki * y + Ri * ubar)
	# 	return u
	# end


	# function bayes_reg_u(A, y, λ₁::T = 1.0, λ₂::T = 0.0) where {T <: Real}
	# 	Qi = I * λ₁
	# 	Ri = I * λ₂
	# 	AQiA = A' * Qi * A
	# 	P = AQiA + Ri
	# 	Ki = AQiA \ (A' * Qi) # atol = pinv_atol
	# 	# Ki = inv(AQiA) * A' * Qi # atol = pinv_atol

	# 	# pre = CholeskyPreconditioner(P,2) 
	# 	if λ₂ == 0
	# 		return Ki * y
	# 	else
	# 		return P \ (AQiA * Ki * y) # 
	# 	end
	# 	# u = cg(P |> CuArray{Float32}, (AQiA * Ki * y) |> CuArray{Float32}) |> Array
	# 	# return u
	# end

	# ho = @thyperopt for i ∈ 4096*2,
	# 	sampler ∈ RandomSampler(), # This is default if none provided
	# 	λ₁ ∈ exp10.(LinRange(-18, 5, 5000)),
	# 	λ₂ ∈ exp10.(LinRange(-18, 5, 5000))

	# 	# f_opti(x, p) = bayes_reg_u(x, Psi, to, λ₁, λ₂)
	# 	# u0 = rand(size(Psi, 2))
	# 	# prob = NonlinearProblem(NonlinearFunction(f_opti), u0, 0.0)
	# 	# sol = solve(prob, NonlinearSolve.LevenbergMarquardt(); maxiters = 10000, abstol = 1e-8)
	# 	# @debug "" sol.retcode
	# 	apc.ExpansionCoefficients .= bayes_reg_u(Psi, to, λ₁, λ₂)
	# 	abs(mean(Psi * apc.ExpansionCoefficients) - mean(to)	)
	# end



	# @info "=> aPC Toolbox: Briefly optimizing the regularization noises for the expansion coefficient solving"
	# f(λ) = mean(abs.(Psi * bayes_reg_u(Psi, to, λ[1], λ[2]) .- to) .^ 2)
	# ho = @hyperopt for resources ∈ 250, sampler ∈ Hyperband(R = 250, η = 5, inner = RandomSampler()),
	# 	λ₁ ∈ [1.0],
	# 	λ₂ ∈ exp10.(LinRange(-13, 1, 500))

	# 	if !(state === nothing)
	# 		λ₁, λ₂ = state
	# 	end
	# 	res = optimize(f, [λ₁, λ₂], NewtonTrustRegion(), Options(time_limit = resources + 1, show_trace = false))
	# 	minimum(res), minimizer(res)
	# end
	# @info ho
	# apc.ExpansionCoefficients .= bayes_reg_u(Psi, to, ho.minimizer[1], ho.minimizer[2])


	# f_opti(x, p) = bayes_reg_u(x, Psi, to, 1.0e-8, 1.0)

	# u0 = rand(size(Psi, 2))
	# prob = NonlinearProblem(NonlinearFunction(f_opti), u0, 0.0)
	# sol = solve(prob, NonlinearSolve.GaussNewton(); maxiters = 10000, abstol = 1e-8, verbose = true)
	# @debug "" sol.retcode
	# apc.ExpansionCoefficients .= bayes_reg_u(Psi, to, 1.0, 0.0001, 0.0)

	# @info size(Psi) size(to) size(Psi'*to)  size((Psi'*Psi))

	# f(α,p) = 0.5*sqrt(sum((Psi'*to .- (Psi'*Psi)*α).^2)) + 1e-9*sum(abs.(α))
	# j(α,p) = ForwardDiff.derivative(α -> f(α,p), α)

	# u0 =  ones(size(Psi,2))

	# prob = NonlinearProblem(NonlinearFunction(f;jac=j), u0, 0.0)
	# sol = solve(prob, NewtonRaphson())
	# @info sol
	# apc.ExpansionCoefficients .= sol.u

	Psi_inv = pinv(Psi)
	apc.ExpansionCoefficients = Psi_inv * to

	# # A is a Matrix and b is a response vector. 
	# Ψ = setupRegularizationProblem(Psi, 2)     # Setup problem
	# b̄, x̄₀ = to_standard_form(Ψ, to, x₀)       # Convert to standard form
	# # Vλ = gcv_tr(Ψ, b̄, x̄₀, 0.1)    
	# cv = Float64[]
	# ls = exp10.(LinRange(-18, 5, 500))   
	# for l in ls                 
	# 	push!(cv,gcv_svd(Ψ, b̄, x̄₀, l))
	# end
	# λ = ls[argmin(cv)]

	# x̄ = solve(Ψ, b̄, x̄₀,λ )                 # Solve the equation
	# x = to_general_form(Ψ, to, x̄)             # Convert back to general form
	# apc.ExpansionCoefficients .= x

	x₀ = apc.ExpansionCoefficients # rand(size(Psi, 1))
	apc.ExpansionCoefficients = invert(Matrix(Psi), to, Lₖx₀(2, x₀); alg = :gcv_svd, method = NelderMead())

	return nothing
end


function predict(apc::aPC{T}, PredictionInput) where {T <: Real}
	@info "=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ..."
	Psi = aPC_PsiPolynomialMatrix(apc, PredictionInput)'
	PredictionOutput = [dot(apc.ExpansionCoefficients, row) for row in eachrow(Psi)]
	return PredictionOutput
end


function UQ(apc::aPC{T}) where {T <: Float64}
	@info "=> aPC Toolbox: UQ Arbitrary Polynomial Chaos ..."
	lc = Array{T}(apc.ExpansionCoefficients)
	OutputMean = Vector{Float64}(lc[1, :])
	OutputVar = Vector{Float64}(sum(lc[2:end, :] .^ 2; dims = 1)[:])
	return (OutputMean = OutputMean, OutputVar = OutputVar)
end


