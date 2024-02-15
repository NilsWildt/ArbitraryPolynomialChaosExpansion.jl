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

mutable struct aPC{T <: Real}
	InputDistribution::RowVecs{T} # in [ d x N-samples]
	input_dimensions::Int64
	ExpansionDegree::Int64
	NumberOfTerms::Int64
	MultivariatePolynomialDegrees::AbstractArray{Int64}
	OrthonormalRepresentation::Bool
	OrthonormalBasis::AbstractArray{T}
	# NumberOfOutputs::Int64
	ExpansionCoefficients::AbstractVector{T}

	# Constructor
	function aPC(
		InputDistribution::RowVecs{T},
		ExpansionDegree::Int64,
		OrthonormalRepresentation::Bool = true,
	) where T
		input_dimensions = size(InputDistribution[1], 1)
		# @info "" size(InputDistribution[1])
		MultivariatePolynomialDegrees = aPC_MultivariatePolynomialDegrees(input_dimensions, ExpansionDegree)
        # display(MultivariatePolynomialDegrees)
		NumberOfTerms = numberPolynomials(ExpansionDegree, input_dimensions)

		OrthonormalBasis = zeros(ExpansionDegree + 2, ExpansionDegree + 2, input_dimensions)
		for i in 1:input_dimensions
			tmp = aPC_OrthonormalBasis(getindex.(InputDistribution, i), ExpansionDegree)
			OrthonormalBasis[:, :, i] .= tmp
		end
		# OrthonormalBasis = tmp

		# display(OrthonormalBasis)
		NumberOfOutputs = 1 # Allocate 100
		ExpansionCoefficients = zeros(T, NumberOfTerms)

		return new{T}(
			InputDistribution,
			input_dimensions,
			ExpansionDegree,
			NumberOfTerms,
			MultivariatePolynomialDegrees,
			OrthonormalRepresentation,
			OrthonormalBasis,
			# NumberOfOutputs,
			ExpansionCoefficients,
		)
	end
end
