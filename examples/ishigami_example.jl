# Ishigami Function Example: Classic UQ Benchmark
# 
# The Ishigami function is a well-known benchmark for uncertainty quantification
# and sensitivity analysis. It exhibits strong nonlinearity and interactions.
#
# f(x1, x2, x3) = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)
# where typically a = 7, b = 0.1
#
# Inputs: x1, x2, x3 ~ Uniform(-pi, pi)

using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion
using Random
using Statistics
using LinearAlgebra
using Printf

# Set random seed for reproducibility
Random.seed!(123)

# -----------------------------------------------------------------------------
# 1. Define the Ishigami Function
# -----------------------------------------------------------------------------

const a = 7.0
const b = 0.1

function ishigami(x::AbstractMatrix)
    x1 = x[:, 1]
    x2 = x[:, 2]
    x3 = x[:, 3]
    return sin.(x1) .+ a .* sin.(x2).^2 .+ b .* x3.^4 .* sin.(x1)
end

# Analytical statistics for Ishigami function (for validation)
# Mean: a/2
# Variance: a^2/8 + b*pi^4/5 + b^2*pi^8/18 + 1/2
analytical_mean = a / 2
analytical_var = a^2/8 + b*pi^4/5 + b^2*pi^8/18 + 1/2

println("Ishigami Function Benchmark")
println("===========================")
println("Parameters: a = $a, b = $b")
println("\nAnalytical Statistics:")
println("  Mean: $(round(analytical_mean, digits=4))")
println("  Variance: $(round(analytical_var, digits=4))")

# -----------------------------------------------------------------------------
# 2. Generate Training and Validation Data
# -----------------------------------------------------------------------------

n_train = 500
n_val = 200
n_dims = 3

# Uniform samples on [-pi, pi]^3
X_train = 2*pi .* rand(n_train, n_dims) .- pi
X_val = 2*pi .* rand(n_val, n_dims) .- pi

Y_train = reshape(ishigami(X_train), :, 1)
Y_val = reshape(ishigami(X_val), :, 1)

println("\nData Generation:")
println("  Training samples: $n_train")
println("  Validation samples: $n_val")

# -----------------------------------------------------------------------------
# 3. Create and Train Polynomial Chaos Expansion
# -----------------------------------------------------------------------------

degree = 8  # Higher degree needed for Ishigami's complexity

println("\nCreating aPCE with degree $degree...")

apc = APCE.aPCE(
    X_train, 
    degree; 
    outdim = 1,
    OrthonormalRepresentation = true,
    center_data = true
)

println("  Number of polynomial terms: $(apc.NumberOfTerms)")

# Train with Bayesian regularization
println("\nTraining model...")
APCE.train!(
    apc, 
    X_train, 
    Y_train; 
    bayesian_inversion = true, 
    reg_order = 2
)

# -----------------------------------------------------------------------------
# 4. Evaluate Predictions
# -----------------------------------------------------------------------------

Y_pred_train = APCE.predict(apc, X_train)
Y_pred_val = APCE.predict(apc, X_val)

# Metrics
train_rmse = sqrt(mean((Y_train .- Y_pred_train).^2))
val_rmse = sqrt(mean((Y_val .- Y_pred_val).^2))
train_r2 = 1 - sum((Y_train .- Y_pred_train).^2) / sum((Y_train .- mean(Y_train)).^2)
val_r2 = 1 - sum((Y_val .- Y_pred_val).^2) / sum((Y_val .- mean(Y_val)).^2)

println("\nPrediction Performance:")
println("  Training RMSE: $(round(train_rmse, digits=4))")
println("  Validation RMSE: $(round(val_rmse, digits=4))")
println("  Training R²: $(round(train_r2, digits=4))")
println("  Validation R²: $(round(val_r2, digits=4))")

# -----------------------------------------------------------------------------
# 5. Uncertainty Quantification
# -----------------------------------------------------------------------------

uq_results = APCE.UQ(apc)

println("\nUncertainty Quantification:")
println("-" ^ 50)
println("  Quantity          | aPCE       | Analytical | Error")
println("-" ^ 50)

mean_error = abs(uq_results.OutputMean[1] - analytical_mean)
var_error = abs(uq_results.OutputVar[1] - analytical_var)

@printf("  Mean              | %10.4f | %10.4f | %10.4e\n", 
        uq_results.OutputMean[1], analytical_mean, mean_error)
@printf("  Variance          | %10.4f | %10.4f | %10.4e\n", 
        uq_results.OutputVar[1], analytical_var, var_error)
println("-" ^ 50)

# Empirical statistics from Monte Carlo (validation data)
emp_mean = mean(Y_val)
emp_var = var(Y_val)

println("\nEmpirical Statistics (Monte Carlo with $n_val samples):")
println("  Empirical Mean: $(round(emp_mean, digits=4))")
println("  Empirical Variance: $(round(emp_var, digits=4))")

# -----------------------------------------------------------------------------
# 6. Summary
# -----------------------------------------------------------------------------

println("\n" * "=" ^ 50)
println("SUMMARY")
println("=" ^ 50)
if val_r2 > 0.99
    println("Excellent fit achieved!")
elseif val_r2 > 0.95
    println("Good fit achieved.")
elseif val_r2 > 0.90
    println("Acceptable fit. Consider increasing polynomial degree.")
else
    println("Poor fit. Try increasing samples or polynomial degree.")
end

println("\nMean prediction error: $(round(100*mean_error/abs(analytical_mean), digits=2))%")
println("Variance prediction error: $(round(100*var_error/analytical_var, digits=2))%")

println("\n--- Ishigami example completed successfully! ---")
