# Basic Example: Polynomial Chaos Expansion
#
# This example demonstrates the core workflow of ArbitraryPolynomialChaosExpansion.jl:
# 1. Generate synthetic training data
# 2. Create a polynomial chaos expansion
# 3. Train the model
# 4. Make predictions
# 5. Perform uncertainty quantification

using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion
using Random
using Statistics
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

# -----------------------------------------------------------------------------
# 1. Generate Synthetic Training Data
# -----------------------------------------------------------------------------
# We use a simple 2D function: f(x1, x2) = sin(x1) + cos(x2) + 0.1*x1*x2

n_samples = 200
n_dims = 2

# Generate input samples (uniform distribution on [0, 1])
TrainingInput = rand(n_samples, n_dims)

# Compute outputs using our target function
target_function(x1, x2) = sin(2 * pi * x1) + cos(2 * pi * x2) + 0.1 * x1 * x2
TrainingOutput = [target_function(TrainingInput[i, 1], TrainingInput[i, 2]) for i in 1:n_samples]
TrainingOutput = reshape(TrainingOutput, :, 1)  # Make it a column matrix

# Split into training and validation sets
split_idx = Int(floor(0.8 * n_samples))
train_idx = 1:split_idx
val_idx = (split_idx + 1):n_samples

X_train = TrainingInput[train_idx, :]
Y_train = TrainingOutput[train_idx, :]
X_val = TrainingInput[val_idx, :]
Y_val = TrainingOutput[val_idx, :]

println("Training samples: $(size(X_train, 1))")
println("Validation samples: $(size(X_val, 1))")

# -----------------------------------------------------------------------------
# 2. Create Polynomial Chaos Expansion
# -----------------------------------------------------------------------------
# The expansion degree controls the complexity of the surrogate model

degree = 5  # Polynomial degree

apc = APCE.aPCE(
    X_train,
    degree;
    outdim = size(Y_train, 2),
    OrthonormalRepresentation = true,
    center_data = true
)

println("\nPolynomial Chaos Expansion created:")
println("  Expansion degree: $degree")
println("  Input dimensions: $n_dims")
println("  Number of terms: $(apc.NumberOfTerms)")

# -----------------------------------------------------------------------------
# 3. Train the Model
# -----------------------------------------------------------------------------
# Using Bayesian regularization for robust coefficient estimation

APCE.train!(
    apc,
    X_train,
    Y_train;
    bayesian_inversion = true,
    reg_order = 2
)

println("\nModel trained successfully!")

# -----------------------------------------------------------------------------
# 4. Make Predictions
# -----------------------------------------------------------------------------

Y_pred_train = APCE.predict(apc, X_train)
Y_pred_val = APCE.predict(apc, X_val)

# Compute errors
train_rmse = sqrt(mean((Y_train .- Y_pred_train) .^ 2))
val_rmse = sqrt(mean((Y_val .- Y_pred_val) .^ 2))
train_r2 = 1 - sum((Y_train .- Y_pred_train) .^ 2) / sum((Y_train .- mean(Y_train)) .^ 2)
val_r2 = 1 - sum((Y_val .- Y_pred_val) .^ 2) / sum((Y_val .- mean(Y_val)) .^ 2)

println("\nPrediction Performance:")
println("  Training RMSE: $(round(train_rmse, digits = 6))")
println("  Validation RMSE: $(round(val_rmse, digits = 6))")
println("  Training R²: $(round(train_r2, digits = 4))")
println("  Validation R²: $(round(val_r2, digits = 4))")

# -----------------------------------------------------------------------------
# 5. Uncertainty Quantification
# -----------------------------------------------------------------------------

uq_results = APCE.UQ(apc)

println("\nUncertainty Quantification Results:")
println("  Output Mean: $(round(uq_results.OutputMean[1], digits = 4))")
println("  Output Variance: $(round(uq_results.OutputVar[1], digits = 4))")
println("  Output Std Dev: $(round(sqrt(uq_results.OutputVar[1]), digits = 4))")

# Compare with empirical statistics from validation data
println("\nEmpirical Statistics (Validation Data):")
println("  Empirical Mean: $(round(mean(Y_val), digits = 4))")
println("  Empirical Variance: $(round(var(Y_val), digits = 4))")
println("  Empirical Std Dev: $(round(std(Y_val), digits = 4))")

println("\n--- Example completed successfully! ---")
