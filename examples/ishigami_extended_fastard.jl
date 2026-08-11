# Extended Ishigami Example: Standard vs Regularized vs FastARD vs Multires+FastARD
#
# The extended Ishigami uses a = 7, b = 0.5 (5× the standard b = 0.1),
# making the x3^4 * sin(x1) interaction much stronger and the problem
# more challenging to approximate with a limited polynomial basis.
#
# We compare four training strategies:
#   1. Standard        — plain pseudoinverse (no regularization), global basis
#   2. Regularized     — Bayesian / Tikhonov regularization, global basis
#   3. FastARD         — sparse Bayesian learning (ARD), global basis
#   4. Multires+FastARD — auto-refined multiresolution basis with per-element
#                         FastARD sparse solves (h-refinement + ARD pruning)
#
# f(x1, x2, x3) = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)
# Inputs: x1, x2, x3 ~ Uniform(-π, π)

using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion
using FastARD
using Random
using Statistics
using LinearAlgebra
using Printf
using CairoMakie

Random.seed!(42)

# -----------------------------------------------------------------------------
# 1. Extended Ishigami Function
# -----------------------------------------------------------------------------

const a_ishi = 7.0
const b_ishi = 0.5  # 5× the classic b = 0.1 (stronger nonlinearity)

function ishigami_extended(x::AbstractMatrix)
    x1 = x[:, 1]
    x2 = x[:, 2]
    x3 = x[:, 3]
    return sin.(x1) .+ a_ishi .* sin.(x2) .^ 2 .+ b_ishi .* x3 .^ 4 .* sin.(x1)
end

# Analytical moments (same formulas, different b)
analytical_mean = a_ishi / 2
analytical_var = a_ishi^2 / 8 + b_ishi * π^4 / 5 + b_ishi^2 * π^8 / 18 + 1 / 2

println("Extended Ishigami Benchmark (a=$a_ishi, b=$b_ishi)")
println("="^50)
@printf("  Analytical Mean:     %10.4f\n", analytical_mean)
@printf("  Analytical Variance: %10.4f\n", analytical_var)

# -----------------------------------------------------------------------------
# 2. Generate Data
# -----------------------------------------------------------------------------

n_train = 300
n_val = 500
n_dims = 3

X_train = 2π .* rand(n_train, n_dims) .- π
X_val = 2π .* rand(n_val, n_dims) .- π

Y_train = reshape(ishigami_extended(X_train), :, 1)
Y_val = reshape(ishigami_extended(X_val), :, 1)

println("\nData: $n_train train / $n_val validation samples")

# -----------------------------------------------------------------------------
# 3. Build the shared global aPCE basis (degree 6 — intentionally moderate)
# -----------------------------------------------------------------------------

degree = 6

apc_std = APCE.aPCE(X_train, degree; outdim = 1, is_orthonormal = true, center_data = true)
apc_reg = APCE.aPCE(X_train, degree; outdim = 1, is_orthonormal = true, center_data = true)
apc_ard = APCE.aPCE(X_train, degree; outdim = 1, is_orthonormal = true, center_data = true)

n_terms = apc_std.NumberOfTerms
println("Polynomial degree: $degree  →  $n_terms basis terms (global)")

# -----------------------------------------------------------------------------
# 4a. Standard solver (no regularization)
# -----------------------------------------------------------------------------

APCE.train!(apc_std, X_train, Y_train; bayesian_inversion = false)

Y_std_train = APCE.predict(apc_std, X_train)
Y_std_val = APCE.predict(apc_std, X_val)

r2_std_train = 1 - sum((Y_train .- Y_std_train) .^ 2) / sum((Y_train .- mean(Y_train)) .^ 2)
r2_std_val = 1 - sum((Y_val .- Y_std_val) .^ 2) / sum((Y_val .- mean(Y_val)) .^ 2)

# -----------------------------------------------------------------------------
# 4b. Regularized solver (Bayesian / Tikhonov)
# -----------------------------------------------------------------------------

APCE.train!(apc_reg, X_train, Y_train; bayesian_inversion = true, reg_order = 2)

Y_reg_train = APCE.predict(apc_reg, X_train)
Y_reg_val = APCE.predict(apc_reg, X_val)

r2_reg_train = 1 - sum((Y_train .- Y_reg_train) .^ 2) / sum((Y_train .- mean(Y_train)) .^ 2)
r2_reg_val = 1 - sum((Y_val .- Y_reg_val) .^ 2) / sum((Y_val .- mean(Y_val)) .^ 2)

# -----------------------------------------------------------------------------
# 4c. FastARD solver (global basis)
# -----------------------------------------------------------------------------

# Get the polynomial design matrix from the shared basis
Psi_train = APCE.aPCE_PsiPolynomialMatrix(apc_ard, X_train)' |> Matrix{Float64}
Psi_val = APCE.aPCE_PsiPolynomialMatrix(apc_ard, X_val)' |> Matrix{Float64}

model_ard = FastARDRegressor(; verbose = false, compute_score = true, n_iter = 300, standardize = false)
fit!(model_ard, Psi_train, vec(Y_train))

active_idx, active_coefs = get_active_coefficients(model_ard)

# Write coefficients back into the aPCE struct for consistent predict / UQ
apc_ard.ExpansionCoefficients .= 0.0
for (i, idx) in enumerate(active_idx)
    apc_ard.ExpansionCoefficients[idx, 1] = active_coefs[i]
end

Y_ard_val, Y_ard_std = predict_with_uncertainty(model_ard, Psi_val)
Y_ard_train_pred = Psi_train * apc_ard.ExpansionCoefficients[:, 1]

r2_ard_train = 1 - sum((vec(Y_train) .- Y_ard_train_pred) .^ 2) / sum((vec(Y_train) .- mean(Y_train)) .^ 2)
r2_ard_val = 1 - sum((vec(Y_val) .- Y_ard_val) .^ 2) / sum((vec(Y_val) .- mean(Y_val)) .^ 2)

n_active = length(active_idx)

# -----------------------------------------------------------------------------
# 4d. Multires + FastARD solver (auto-refined multiresolution basis)
# -----------------------------------------------------------------------------

# Build a multiresolution basis (starts as a single element covering the full
# domain), then let auto_refine! split it where the response has the most
# between-group variance — with FastARD doing the per-element sparse solve at
# every refinement step.
apc_mr = APCE.aPCE(X_train, degree; basis = Val(:multires))

println("\nRunning auto_refine! with solver = :fastard ...")
APCE.auto_refine!(apc_mr, X_train, Y_train; max_elements = 8, solver = :fastard)

Y_mr_train = APCE.predict(apc_mr, X_train)
Y_mr_val = APCE.predict(apc_mr, X_val)

r2_mr_train = 1 - sum((Y_train .- Y_mr_train) .^ 2) / sum((Y_train .- mean(Y_train)) .^ 2)
r2_mr_val = 1 - sum((Y_val .- Y_mr_val) .^ 2) / sum((Y_val .- mean(Y_val)) .^ 2)

n_elements = length(apc_mr.OrthonormalBasis.elements)
# Count non-zero per-element coefficients (across all elements)
n_nonzero_mr = sum(
    count(!iszero, @view elem.coefficients[:, 1]) for
    elem in apc_mr.OrthonormalBasis.elements if !isempty(elem.coefficients)
)
n_total_mr = sum(
    size(elem.coefficients, 1) for
    elem in apc_mr.OrthonormalBasis.elements if !isempty(elem.coefficients)
)

println("  Multires elements after auto-refine: $n_elements")
println("  Non-zero local coefficients: $n_nonzero_mr / $n_total_mr")

# -----------------------------------------------------------------------------
# 5. Print comparison table
# -----------------------------------------------------------------------------

println("\n" * "="^78)
println("  Method               | R² (train) | R² (val)  | Active / Total terms")
println("-"^78)
@printf("  Standard             | %10.4f | %9.4f | %d / %d\n", r2_std_train, r2_std_val, n_terms, n_terms)
@printf("  Regularized          | %10.4f | %9.4f | %d / %d\n", r2_reg_train, r2_reg_val, n_terms, n_terms)
@printf("  FastARD              | %10.4f | %9.4f | %d / %d\n", r2_ard_train, r2_ard_val, n_active, n_terms)
@printf("  Multires+FastARD     | %10.4f | %9.4f | %d / %d (%d elem.)\n",
    r2_mr_train, r2_mr_val, n_nonzero_mr, n_total_mr, n_elements)
println("="^78)

# -----------------------------------------------------------------------------
# 6. Plot: Four-panel prediction-vs-data comparison
# -----------------------------------------------------------------------------

function r2_label(r2)
    return "R² = $(round(r2, digits = 4))"
end

fig = Figure(size = (2000, 480))

methods = [
    ("Standard (pinv)", Y_std_val, r2_std_val, :coral,
        "$n_terms / $n_terms terms"),
    ("Regularized (Bayesian)", Y_reg_val, r2_reg_val, :mediumseagreen,
        "$n_terms / $n_terms terms"),
    ("FastARD", Y_ard_val, r2_ard_val, :dodgerblue,
        "$n_active / $n_terms terms"),
    ("Multires + FastARD", Y_mr_val, r2_mr_val, :darkorange,
        "$n_nonzero_mr / $n_total_mr ($n_elements elem.)"),
]

for (i, (label, Y_pred, r2, col, sub)) in enumerate(methods)
    ax = Axis(
        fig[1, i];
        xlabel = "True Response",
        ylabel = i == 1 ? "Predicted" : "",
        title = "$label\n$(r2_label(r2))  ·  $sub",
        aspect = DataAspect(),
    )

    dmin, dmax = extrema(Y_val)
    lines!(ax, [dmin, dmax], [dmin, dmax]; color = :grey60, linewidth = 1.5, linestyle = :dash)
    scatter!(ax, vec(Y_val), vec(Y_pred); color = (col, 0.45), markersize = 5)
end

save(joinpath(@__DIR__, "..", "figures", "ishigami_extended_comparison.png"), fig; px_per_unit = 3)
println("\nFigure saved to figures/ishigami_extended_comparison.png")
println("\n--- Extended Ishigami + FastARD + Multires example completed successfully! ---")
