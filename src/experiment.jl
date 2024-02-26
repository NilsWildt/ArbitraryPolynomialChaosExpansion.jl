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
using Random
import Distributions: Beta, Uniform

export get_input
export PhysicalModelND

function get_input(N, d, seed = 1)
	rng = Xoshiro(seed)
	if d == 2
		x = [rand(rng, Beta(2, 1), N) rand(rng, Uniform(0, 1), N)]
		return RowVecs(x)
	elseif d == 1
		x = rand(rng, Beta(2, 1), N)
		return RowVecs(x)
	else
		@error "Currently only d==1 and d==2 is implemented."
	end
end

function add_zero_dimension(P::AbstractArray)
	# Create a zero array with the same size as P
	zero_array = zeros(size(P))

	# Concatenate P and zero_array along the first dimension
	P_new = vcat(P, zero_array)

	return P_new
end

function PhysicalModelND(t, P::AbstractArray)
	ModelResponse = (P[1]^2 + P[2] - 1.0) .^ 2 .+ P[1]^3 + 0.5 * P[1] * exp(P[2]) .- sqrt.(t) .* P[1]
	for i ∈ 3:lastindex(P, 1)
		ModelResponse .+= P[i]
	end
	return ModelResponse .+ rand() .* 0.1 # SVector{length(ModelResponse)}(
end
# function PhysicalModelND(t, P::AbstractArray)
# 	# @debug "Assuming a nd case" P
# 	# P = reduce(hcat, P)
# 	# P = add_zero_dimension(P)
# 	ModelResponse = 4.0*((P[1]-0.5)^2+(P[2]-0.5)^2)

# 	# for i ∈ 3:lastindex(P, 1)
# 	# 	ModelResponse .+= P[i]
# 	# end
# 	return ModelResponse #.+rand() .*0.1 # SVector{length(ModelResponse)}(
# end


function PhysicalModel1D(t, P)
	ModelResponse = @. (P[1]^2 + 0.0 - 1.0) .^ 2 .+ P[1]^3 + 0.5 * P[1] * exp(0.0) .- sqrt.(t) .* P[1]

	for i ∈ 3:lastindex(P, 1)
		ModelResponse .+= P[i]
	end

	return ModelResponse .+ rand() * 0.3 # SVector{length(ModelResponse)}(
end


function PhysicalModel2DGaussian(t, P::AbstractArray)
	# Ensure P is a 2D array
	# P = reduce(hcat, P)

	# Check that P has two dimensions
	if size(P, 1) != 2
		@error "P must have two dimensions for the 2D Gaussian model."
	end

	# Parameters for the 2D Gaussian
	mu_x = 0
	mu_y = 0
	sigma_x = 1
	sigma_y = 1

	# Calculate the 2D Gaussian
	ModelResponse = exp(-((P[1] - mu_x)^2 / (2 * sigma_x^2) + (P[2] - mu_y)^2 / (2 * sigma_y^2)))

	return ModelResponse
end
