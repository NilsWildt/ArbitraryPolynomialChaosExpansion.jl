using BenchmarkTools
using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion
using Random
using LinearAlgebra

const SUITE = BenchmarkGroup()

# Use fixed seed for reproducible benchmarks
const RNG = Random.Xoshiro(42)

# =============================================================================
# Test Functions
# =============================================================================

# Simple nonlinear function for reliable benchmarks (works on [0,1] data)
simple_nonlinear(x) = sin.(x[:, 1]) .+ cos.(x[:, 2]) .+ 0.1 .* x[:, 1] .* x[:, 2]

# Ishigami function - classic UQ benchmark (works on [-pi, pi] data)
const ISHIGAMI_A = 7.0
const ISHIGAMI_B = 0.1
function ishigami(x::AbstractMatrix)
    return sin.(x[:, 1]) .+ ISHIGAMI_A .* sin.(x[:, 2]).^2 .+ ISHIGAMI_B .* x[:, 3].^4 .* sin.(x[:, 1])
end

# =============================================================================
# 1. BASIS CONSTRUCTION BENCHMARKS
# =============================================================================
# Core computational kernels - orthonormal basis construction

SUITE["basis"] = BenchmarkGroup()

# 2D basis - various sizes and degrees
for (n_samples, degree) in [(100, 3), (200, 4), (500, 4), (1000, 3)]
    data = rand(RNG, n_samples, 2)
    SUITE["basis"]["ortho_$(n_samples)x2_deg$(degree)"] = @benchmarkable(
        APCE.create_basis($data, $degree; center_data=true),
        samples=30, evals=1
    )
end

# 3D basis
for (n_samples, degree) in [(200, 3), (300, 4)]
    data = rand(RNG, n_samples, 3)
    SUITE["basis"]["ortho_$(n_samples)x3_deg$(degree)"] = @benchmarkable(
        APCE.create_basis($data, $degree; center_data=true),
        samples=20, evals=1
    )
end

# Full (non-orthonormal) basis for comparison
data_full = rand(RNG, 200, 2)
SUITE["basis"]["full_200x2_deg4"] = @benchmarkable(
    APCE.create_basis($data_full, 4, Val(false)),
    samples=30, evals=1
)

# =============================================================================
# 2. aPCE CONSTRUCTION BENCHMARKS  
# =============================================================================
# Full aPCE object construction

SUITE["construct"] = BenchmarkGroup()

for (n_samples, n_dims, degree) in [(100, 2, 3), (200, 2, 4), (300, 2, 5), (200, 3, 3)]
    data = rand(RNG, n_samples, n_dims)
    SUITE["construct"]["aPCE_$(n_samples)x$(n_dims)_deg$(degree)"] = @benchmarkable(
        APCE.aPCE($data, $degree; outdim=1, OrthonormalRepresentation=true),
        samples=15, evals=1
    )
end

# Multi-output construction
data_mo = rand(RNG, 200, 2)
SUITE["construct"]["aPCE_multioutput_200x2_deg3_out3"] = @benchmarkable(
    APCE.aPCE($data_mo, 3; outdim=3, OrthonormalRepresentation=true),
    samples=10, evals=1
)

# =============================================================================
# 3. TRAINING BENCHMARKS
# =============================================================================
# Training is often the computational bottleneck

SUITE["train"] = BenchmarkGroup()

# Basic least squares training (fast)
for (n_samples, degree) in [(100, 3), (200, 4), (300, 3)]
    X = rand(RNG, n_samples, 2)
    Y = reshape(simple_nonlinear(X), :, 1)
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    
    SUITE["train"]["basic_$(n_samples)x2_deg$(degree)"] = @benchmarkable(
        APCE.train!(apc_copy, $X, $Y),
        samples=20, evals=1,
        setup=(apc_copy = deepcopy($apc))
    )
end

# Bayesian regularization - most common in practice
# Use well-conditioned setups (more samples than polynomial terms)
for (n_samples, degree, reg_order) in [(150, 3, 0), (150, 3, 2), (200, 4, 2)]
    X = rand(RNG, n_samples, 2)
    Y = reshape(simple_nonlinear(X), :, 1)
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    
    SUITE["train"]["bayesian_$(n_samples)x2_deg$(degree)_reg$(reg_order)"] = @benchmarkable(
        APCE.train!(apc_copy, $X, $Y; bayesian_inversion=true, reg_order=$reg_order),
        samples=10, evals=1,
        setup=(apc_copy = deepcopy($apc))
    )
end

# Multi-output training
X_mo = rand(RNG, 150, 2)
Y_mo = hcat(simple_nonlinear(X_mo), sin.(X_mo[:, 1]) .* X_mo[:, 2])
apc_mo = APCE.aPCE(X_mo, 3; outdim=2, OrthonormalRepresentation=true)

SUITE["train"]["multioutput_150x2_deg3_out2"] = @benchmarkable(
    APCE.train!(apc_copy, $X_mo, $Y_mo; bayesian_inversion=true),
    samples=10, evals=1,
    setup=(apc_copy = deepcopy($apc_mo))
)

# =============================================================================
# 4. PREDICTION BENCHMARKS
# =============================================================================
# Prediction speed is critical for surrogate model applications

SUITE["predict"] = BenchmarkGroup()

# Setup trained models using basic training (reliable, fast setup)
function make_trained_model_2d(n_train, degree)
    X = rand(RNG, n_train, 2)
    Y = reshape(simple_nonlinear(X), :, 1)
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    APCE.train!(apc, X, Y)  # Basic training - always works
    return apc
end

# 2D predictions - various sizes
apc_2d_deg3 = make_trained_model_2d(200, 3)
apc_2d_deg4 = make_trained_model_2d(200, 4)

for n_pred in [100, 500, 1000]
    X_test = rand(RNG, n_pred, 2)
    SUITE["predict"]["2d_deg3_$(n_pred)pts"] = @benchmarkable(
        APCE.predict($apc_2d_deg3, $X_test),
        samples=50, evals=1
    )
end

for n_pred in [100, 500]
    X_test = rand(RNG, n_pred, 2)
    SUITE["predict"]["2d_deg4_$(n_pred)pts"] = @benchmarkable(
        APCE.predict($apc_2d_deg4, $X_test),
        samples=50, evals=1
    )
end

# =============================================================================
# 5. PSI MATRIX EVALUATION BENCHMARKS
# =============================================================================
# Core operation used in both training and prediction

SUITE["psi_matrix"] = BenchmarkGroup()

for (n_samples, n_dims, degree) in [(100, 2, 3), (200, 2, 4), (500, 2, 3), (200, 3, 3)]
    X = rand(RNG, n_samples, n_dims)
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    
    SUITE["psi_matrix"]["$(n_samples)x$(n_dims)_deg$(degree)"] = @benchmarkable(
        APCE.aPCE_PsiPolynomialMatrix($apc, $X),
        samples=30, evals=1
    )
end

# =============================================================================
# 6. UNCERTAINTY QUANTIFICATION BENCHMARKS
# =============================================================================
# UQ is the primary use case for PCE

SUITE["uq"] = BenchmarkGroup()

for degree in [3, 4, 5]
    X = rand(RNG, 200, 2)
    Y = reshape(simple_nonlinear(X), :, 1)
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    APCE.train!(apc, X, Y)
    
    SUITE["uq"]["2d_deg$(degree)"] = @benchmarkable(
        APCE.UQ($apc),
        samples=100, evals=1
    )
end

# Multi-output UQ
apc_uq_mo = APCE.aPCE(rand(RNG, 150, 2), 3; outdim=2, OrthonormalRepresentation=true)
APCE.train!(apc_uq_mo, rand(RNG, 150, 2), rand(RNG, 150, 2))

SUITE["uq"]["multioutput_2d_deg3_out2"] = @benchmarkable(
    APCE.UQ($apc_uq_mo),
    samples=50, evals=1
)

# =============================================================================
# 7. END-TO-END WORKFLOW BENCHMARKS
# =============================================================================
# Complete realistic workflows

SUITE["workflow"] = BenchmarkGroup()

# Simple 2D workflow - the most common use case
function workflow_2d_basic(n_samples, degree)
    X = rand(n_samples, 2)
    Y = reshape(simple_nonlinear(X), :, 1)
    
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    APCE.train!(apc, X, Y)
    
    X_test = rand(50, 2)
    pred = APCE.predict(apc, X_test)
    uq = APCE.UQ(apc)
    
    return uq
end

SUITE["workflow"]["basic_2d_150_deg3"] = @benchmarkable(
    workflow_2d_basic(150, 3),
    samples=10, evals=1
)

SUITE["workflow"]["basic_2d_200_deg4"] = @benchmarkable(
    workflow_2d_basic(200, 4),
    samples=10, evals=1
)

# Workflow with Bayesian regularization
function workflow_2d_bayesian(n_samples, degree)
    X = rand(n_samples, 2)
    Y = reshape(simple_nonlinear(X), :, 1)
    
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    APCE.train!(apc, X, Y; bayesian_inversion=true, reg_order=2)
    
    X_test = rand(50, 2)
    pred = APCE.predict(apc, X_test)
    uq = APCE.UQ(apc)
    
    return uq
end

SUITE["workflow"]["bayesian_2d_150_deg3"] = @benchmarkable(
    workflow_2d_bayesian(150, 3),
    samples=5, evals=1
)

# Ishigami function workflow - classic UQ benchmark
function workflow_ishigami(n_samples, degree)
    X = 2*pi .* rand(n_samples, 3) .- pi  # Uniform on [-pi, pi]
    Y = reshape(ishigami(X), :, 1)
    
    apc = APCE.aPCE(X, degree; outdim=1, OrthonormalRepresentation=true)
    APCE.train!(apc, X, Y)  # Basic training for reliability
    
    X_test = 2*pi .* rand(100, 3) .- pi
    pred = APCE.predict(apc, X_test)
    uq = APCE.UQ(apc)
    
    return uq
end

SUITE["workflow"]["ishigami_200_deg3"] = @benchmarkable(
    workflow_ishigami(200, 3),
    samples=5, evals=1
)

SUITE["workflow"]["ishigami_300_deg4"] = @benchmarkable(
    workflow_ishigami(300, 4),
    samples=3, evals=1
)
