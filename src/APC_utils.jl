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

function numberPolynomials(n::Int64,d::Int64)
    x, y = max(d, n), min(d, n)
    return UInt128(prod(UInt128(x + 1):UInt128(d + n)) ÷ factorial(UInt128(y))) |>Int
end


function reverse_columns!(x)
    for row in axes(x, 1)
        x[row, :] = reverse(x[row, :])
    end
end


function train!(apc::aPC{T},TrainingInput::RowVecs,TrainingOutput::RowVecs) where {T<:Real}
    # @info "Training the arbitrary Polynomial Chaos"
# @warn "" size(TrainingInput)
    Psi = aPC_PsiPolynomialMatrix(apc,TrainingInput)'
    to = reduce(vcat,TrainingOutput)
    # @debug "" size(Psi) Psi to size(to) TrainingOutput
# 	# if  apc.input_dimensions == 1
        # Psi_inv = pinv(Psi) 
    
# 	# @debug ""  Psi_inv TrainingOutput
    # apc.ExpansionCoefficients = Psi_inv*to
# display(apc.ExpansionCoefficients )
# else
        Psi_inv = pinv(Psi)
    # @debug ""  Psi_inv to
    apc.ExpansionCoefficients = Psi_inv*to
    # apc.ExpansionCoefficients = Psi\to
# end
     return nothing
end


function predict(apc::aPC{T}, PredictionInput) where {T<:Real}
    println("=> aPC Toolbox: Prediction using Arbitrary Polynomial Chaos ...")
    
    # Assuming aPC_PsiPolynomialMatrix is correctly implemented in Julia as discussed before
    Psi = aPC_PsiPolynomialMatrix(apc, PredictionInput)'
    # Initialize the prediction output matrix
	# @debug "" Psi apc.ExpansionCoefficients
    # PredictionOutput = zeros(T, size(PredictionInput, 1))
    # Perform prediction using the expansion coefficients
    # PredictionOutput = Psi * apc.ExpansionCoefficients
    PredictionOutput = [dot(apc.ExpansionCoefficients,row) for row in eachrow(Psi)]
    return PredictionOutput
end


function UQ(apc::aPC{T}) where {T<:Float64}
	lc = Array{T}(copy(apc.ExpansionCoefficients))
    OutputMean = Vector{Float64}(lc[1, :])
	OutputVar = Vector{Float64}(sum(lc[2:end, :].^2; dims=1)[:])
    return (OutputMean=OutputMean, OutputVar=OutputVar)
end