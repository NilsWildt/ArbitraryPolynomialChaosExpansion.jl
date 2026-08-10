# multires_benchmark.jl
# Generates figures/multires_bimodal_comparison.png
#
# Demonstrates that a single global polynomial basis struggles with a
# discontinuous (step) response, while the multiresolution basis — which
# splits the domain at the discontinuity — captures it cleanly.

using ArbitraryPolynomialChaosExpansion
const APCE = ArbitraryPolynomialChaosExpansion
using Random
using Statistics
using Plots
gr()

# ---------------------------------------------------------------------------
# 1-D step function: f(x) = -1 for x < 0.5, +1 for x >= 0.5
# ---------------------------------------------------------------------------
f_true(x) = x < 0.5 ? -1.0 : 1.0

Random.seed!(42)
N_train = 80
X_train = reshape(sort(rand(N_train)), :, 1)
y_train = f_true.(X_train[:, 1]) .+ 0.05 .* randn(N_train)  # small noise

degree = 6

# ---------------------------------------------------------------------------
# Global aPCE (single element)
# ---------------------------------------------------------------------------
apc_global = APCE.aPCE(X_train, degree; basis = Val(:recurrence))
APCE.train!(apc_global, X_train, y_train)

# ---------------------------------------------------------------------------
# Multires aPCE (split at x = 0.5 — the discontinuity)
# ---------------------------------------------------------------------------
apc_mr = APCE.aPCE(X_train, degree; basis = Val(:multires),
                   split_dim = 1, split_point = 0.5)
APCE.train!(apc_mr, X_train, y_train)

# ---------------------------------------------------------------------------
# Dense prediction grid
# ---------------------------------------------------------------------------
X_fine = reshape(collect(0.0:0.002:1.0), :, 1)
y_truth = f_true.(X_fine[:, 1])
y_global = APCE.predict(apc_global, X_fine)[:, 1]
y_mr = APCE.predict(apc_mr, X_fine)[:, 1]

# MSE on fine grid (excluding immediate vicinity of discontinuity)
mask = abs.(X_fine[:, 1] .- 0.5) .> 0.02
mse_global = mean((y_global[mask] .- y_truth[mask]) .^ 2)
mse_mr = mean((y_mr[mask] .- y_truth[mask]) .^ 2)

println("Global  MSE: $(round(mse_global, sigdigits=4))")
println("MultiRes MSE: $(round(mse_mr, sigdigits=4))")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
p = plot(X_fine[:, 1], y_truth,
         label = "True function",
         color = :black, linewidth = 2, linestyle = :dash,
         xlabel = "x", ylabel = "f(x)",
         title = "Multiresolution aPCE vs Global aPCE (degree $degree)",
         legend = :topright, size = (800, 500),
         margins = 5Plots.mm)

plot!(p, X_fine[:, 1], y_global,
      label = "Global aPCE  (MSE = $(round(mse_global, sigdigits=3)))",
      color = :steelblue, linewidth = 2, alpha = 0.9)

plot!(p, X_fine[:, 1], y_mr,
      label = "Multires aPCE  (MSE = $(round(mse_mr, sigdigits=3)))",
      color = :crimson, linewidth = 2, alpha = 0.9)

scatter!(p, X_train[:, 1], y_train,
         label = "Training points",
         color = :gray, markersize = 2, alpha = 0.5)

vline!(p, [0.5], color = :darkgray, linestyle = :dot, label = "Split point")

savefig(p, joinpath(@__DIR__, "..", "figures", "multires_bimodal_comparison.png"))
println("Saved figures/multires_bimodal_comparison.png")
