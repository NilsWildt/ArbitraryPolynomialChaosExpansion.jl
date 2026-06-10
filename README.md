# ArbitraryPolynomialChaosExpansion.jl

[![CI](https://github.com/NilsWildt/ArbitraryPolynomialChaosExpansion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/NilsWildt/ArbitraryPolynomialChaosExpansion.jl/actions/workflows/CI.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Registry: unregistered](https://img.shields.io/badge/registry-unregistered-lightgrey.svg)](#installation)

> **Note:** This package is **not yet registered** in the Julia General registry.
> Install it directly from GitHub — see [Installation](#installation).

A Julia package for constructing polynomial chaos expansions with arbitrary polynomial bases, enabling uncertainty quantification and surrogate modeling for complex systems.

## Overview

ArbitraryPolynomialChaosExpansion.jl (APCE) provides efficient tools for:
- **Arbitrary Polynomial Basis Construction**: Create orthonormal or full polynomial bases from data
- **Moment-Based Orthogonalization**: Construct data-driven orthogonal polynomial bases
- **Uncertainty Quantification**: Compute statistical moments and sensitivities
- **Surrogate Modeling**: Build fast-to-evaluate polynomial approximations of expensive simulations
## Features

- Data-driven orthonormal polynomial basis construction (degrees 0-4 with closed-form solutions)
- Support for multivariate polynomial expansions with flexible term selection
- Bayesian regularization for robust coefficient estimation
- Gaussian collocation point selection (Probabilistic Collocation Method)
- Type-stable implementations optimized for performance
- Comprehensive automatic differentiation support via ChainRules
- Extensive test coverage with TestItems

## Installation

This package is not in the General registry yet, so install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/NilsWildt/ArbitraryPolynomialChaosExpansion.jl")
```

Or from the Julia REPL package mode:
```julia
] add https://github.com/NilsWildt/ArbitraryPolynomialChaosExpansion.jl
```

## Quick Start

```julia
using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion

# Generate training data
TrainingInput = rand(100, 2)  # 100 samples, 2 dimensions
TrainingOutput = sin.(TrainingInput[:, 1]) .* cos.(TrainingInput[:, 2])

# Create polynomial chaos expansion
degree = 3
apc = aPCE(TrainingInput, degree; outdim=1, OrthonormalRepresentation=true)

# Train the model
train!(apc, TrainingInput, TrainingOutput; bayesian_inversion=true, reg_order=2)

# Make predictions
TestInput = rand(20, 2)
predictions = predict(apc, TestInput)

# Uncertainty quantification
uq_results = UQ(apc)
println("Output Mean: ", uq_results.OutputMean)
println("Output Variance: ", uq_results.OutputVar)
```

## Results

### Ishigami Function Benchmark

The classic Ishigami function (a = 7, b = 0.1) with 500 training samples and degree-8 expansion achieves near-perfect surrogate accuracy on held-out validation data:

<p align="center">
  <img src="figures/ishigami_prediction.png" width="500" alt="Ishigami prediction vs true response">
</p>

### Extended Ishigami: Standard vs Regularized vs FastARD

An extended variant (b = 0.5, 5× the standard) with only 300 training samples and a moderate degree-6 basis (84 terms) exposes the difference between solvers. [FastARD.jl](https://github.com/NilsWildt/FastARD.jl) automatically prunes irrelevant basis functions, achieving the best validation R² with a sparse solution:

| Method | R² (val) | Active / Total terms |
|--------|----------|----------------------|
| Standard (pinv) | 0.9449 | 84 / 84 |
| Regularized (Bayesian) | 0.9449 | 84 / 84 |
| **FastARD** | **0.9636** | **53 / 84** |

<p align="center">
  <img src="figures/ishigami_extended_comparison.png" width="100%" alt="Extended Ishigami comparison: Standard vs Regularized vs FastARD">
</p>

## Usage

### Creating Polynomial Bases

```julia
# Create orthonormal basis (recommended)
x = rand(100, 3)
degree = 4
basis_ortho = create_basis(x, degree; center_data=true)

# Create full/monomial basis
basis_full = create_basis(x, degree, Val(false))
```

### Training with Regularization

```julia
# Basic training
train!(apc, TrainingInput, TrainingOutput)

# Training with Bayesian regularization (recommended for ill-conditioned problems)
train!(apc, TrainingInput, TrainingOutput; 
       bayesian_inversion=true, 
       reg_order=2)  # 0, 1, 2, or 3 for different regularization orders
```

### Advanced: Gaussian Collocation

```julia
# Enable Gaussian collocation during construction
apc = aPCE(TrainingInput, degree; do_gauss=true, outdim=1)

# Generate collocation points
collocation_points = GaussianCollocation(apc; strategy=:PCM)
```

### Customizing Polynomial Terms

```julia
# Control marginal and interaction terms
apc = aPCE(TrainingInput, degree;
           s_marginals=1.0,      # Keep all marginal terms
           s_interactions=0.5,   # Keep 50% of interaction terms
           outdim=1)
```

## API Documentation

### Main Types

- `aPCE`: Main struct for polynomial chaos expansion

### Core Functions

- `aPCE(InputDistribution, ExpansionDegree; ...)`: Constructor for polynomial chaos expansion
- `train!(apc, TrainingInput, TrainingOutput; ...)`: Train expansion coefficients
- `predict(apc, PredictionInput)`: Evaluate expansion at new points
- `UQ(apc)`: Compute uncertainty quantification statistics

### Basis Functions

- `create_basis(x, degree; center_data=true)`: Create polynomial basis
- `aPCE_OrthonormalBasis(Data, Degree, center_data)`: Construct orthonormal basis
- `aPCE_MultivariatePolynomialDegrees(num_dimensions, max_degree, s_marginals, s_interactions)`: Generate polynomial degree indices

### Collocation

- `GaussianCollocation(apc; strategy=:PCM)`: Generate Gaussian quadrature points

## Examples

See the `examples/` directory for comprehensive usage examples:
- `basic_example.jl`: Basic workflow demonstrating the core API
- `ishigami_example.jl`: Classic Ishigami function UQ benchmark with analytical validation
- `ishigami_extended_fastard.jl`: Extended Ishigami variant comparing Standard, Regularized, and [FastARD](https://github.com/NilsWildt/FastARD.jl) solvers

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Polynomial Chaos theory and applications in uncertainty quantification
- Arbitrary Polynomial Chaos methodology for data-driven bases
- Bayesian regularization for inverse problems

## Acknowledgments

This package builds upon research in uncertainty quantification and polynomial chaos methods.
