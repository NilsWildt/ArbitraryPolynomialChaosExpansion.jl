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

	# Psi_inv = pinv(Psi)
	# apc.ExpansionCoefficients = Psi_inv*to

	## See p. 96 Sullivan UQ 

	function ℓπ(u, A, y, λ₁::T = 1.0e-8, λ₂::T = 1.0) where {T <: Real}
		AQiA = A' * (I * 1.0./λ₁) * A
		P = AQiA + I * 1.0./λ₁
		K = pinv(AQiA) * A' * I * 1.0./λ₁
		[ -(-0.5 * sum((u .- inv(P) * (AQiA * K * y + I * 1.0./λ₂ * u)) .^ 2))]
	end

	f_opti(x, p) = ℓπ(x, Psi, to)
	u0 = rand(size(Psi, 2))
	prob = NonlinearProblem(NonlinearFunction(f_opti), u0, 0.0)
	sol = solve(prob, NonlinearSolve.NewtonRaphson())
    @info "" sol.retcode
	apc.ExpansionCoefficients .= sol.u[:, :]

	# @info size(Psi) size(to) size(Psi'*to)  size((Psi'*Psi))

	# f(α,p) = 0.5*sqrt(sum((Psi'*to .- (Psi'*Psi)*α).^2)) + 1e-9*sum(abs.(α))
	# j(α,p) = ForwardDiff.derivative(α -> f(α,p), α)

	# u0 =  ones(size(Psi,2))

	# prob = NonlinearProblem(NonlinearFunction(f;jac=j), u0, 0.0)
	# sol = solve(prob, NewtonRaphson())
	# @info sol
	# apc.ExpansionCoefficients .= sol.u
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
