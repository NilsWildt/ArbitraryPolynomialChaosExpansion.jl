using LinearAlgebra
using Statistics
using Printf
using TimerOutputs
const to = TimerOutput()
# ==============================================================================
# VERSION 1: Hankel Matrix Method (Oladyshkin & Nowak, 2012)
# ==============================================================================

"""
    aPCE_Hankel(Data, Degree)

Version 1: Constructs basis using explicit Moment Matrix inversion (Hankel solve).
"""
function aPCE_Hankel(Data::AbstractArray{T}, Degree::Integer, center::Bool=true) where {T<:Real}
    x_raw = vec(Data)

    if center
        μ = mean(x_raw)
        σ_val = std(x_raw; mean=μ)
        σ = σ_val > eps(T) ? σ_val : one(T)
        x = (x_raw .- μ) ./ σ
    else
        x = x_raw
        μ, σ = zero(T), one(T)
    end

    # Build basis in transformed space
    OrthonormalCoeffs = _v1_build_orthonormal_basis(x, Degree)

    if center
        return _v1_backtransform(OrthonormalCoeffs, μ, σ, Degree)
    else
        return OrthonormalCoeffs
    end
end

function _v1_build_orthonormal_basis(x::AbstractVector{T}, D::Integer) where {T<:Real}
    # Compute moments: m[k+1] = E[x^k]
    m = zeros(T, 2D + 1)
    for k in 0:2D
        m[k+1] = mean(x .^ k)
    end

    # Monic orthogonal polynomials via Hankel solve (Eq. 14)
    MonicCoeffs = zeros(T, D + 1, D + 1)
    MonicCoeffs[1, 1] = one(T)  # P_0 = 1

    for k in 1:D
        # Build moment matrix
        M = zeros(T, k + 1, k + 1)
        rhs = zeros(T, k + 1)

        for row in 0:(k-1)
            for col in 0:k
                M[row+1, col+1] = m[row+col+1]
            end
        end

        # Monic constraint: p_k = 1
        M[k+1, k+1] = one(T)
        rhs[k+1] = one(T)

        # Potential singularity warning for high degrees
        try
            MonicCoeffs[k+1, 1:(k+1)] = M \ rhs
        catch e
            @warn "V1 (Hankel) Singular Matrix at degree $k. Returning zeros."
            return MonicCoeffs
        end
    end

    # Normalize via Hankel inner product
    OrthonormalCoeffs = zeros(T, D + 1, D + 1)
    for k in 0:D
        norm_sq = zero(T)
        for i in 0:k
            for j in 0:k
                norm_sq += MonicCoeffs[k+1, i+1] * MonicCoeffs[k+1, j+1] * m[i+j+1]
            end
        end
        norm_factor = sqrt(max(norm_sq, eps(T)))
        OrthonormalCoeffs[k+1, 1:(k+1)] = MonicCoeffs[k+1, 1:(k+1)] ./ norm_factor
    end

    return OrthonormalCoeffs
end

function _v1_backtransform(Coeffs::AbstractMatrix{T}, μ::T, σ::T, D::Integer) where {T<:Real}
    Result = zeros(T, D + 1, D + 1)
    for k in 0:D
        for j in 0:k
            c = Coeffs[k+1, j+1]
            inv_σ_j = one(T) / (σ^j)
            for l in 0:j
                Result[k+1, l+1] += c * inv_σ_j * binomial(j, l) * ((-μ)^(j - l))
            end
        end
    end
    return Result
end

# ==============================================================================
# VERSION 2: Stieltjes Procedure
# ==============================================================================

"""
    aPCE_Stieltjes(Data, Degree)

Version 2: Constructs basis using Stieltjes recurrence (Stable).
"""
function aPCE_Stieltjes(Data::AbstractArray{T}, Degree::Integer, center::Bool=true) where {T<:Real}
    x_raw = vec(Data)
    D = Degree

    if center
        μ = mean(x_raw)
        σ_val = std(x_raw; mean=μ)
        σ = σ_val > eps(T) ? σ_val : one(T)
        x = (x_raw .- μ) ./ σ
    else
        x = x_raw
        μ, σ = zero(T), one(T)
    end

    m, MonicCoeffs = _v2_stieltjes_core(x, D)
    OrthonormalCoeffs = _v2_normalize_hankel(MonicCoeffs, m, D)

    if center
        FinalBasis = zeros(T, D + 1, D + 1)
        for k in 0:D
            for j in 0:k
                c = OrthonormalCoeffs[k+1, j+1]
                inv_σ_j = one(T) / (σ^j)
                for l in 0:j
                    FinalBasis[k+1, l+1] += c * inv_σ_j * binomial(j, l) * ((-μ)^(j - l))
                end
            end
        end
        return FinalBasis
    else
        return OrthonormalCoeffs
    end
end

function _v2_stieltjes_core(x::AbstractVector{T}, D::Integer) where {T<:Real}
    N = length(x)
    m = zeros(T, 2D + 1)
    for k in 0:2D
        m[k+1] = mean(x .^ k)
    end

    P_prev = zeros(T, N)
    P_curr = ones(T, N)
    MonicCoeffs = zeros(T, D + 1, D + 1)
    MonicCoeffs[1, 1] = one(T)
    inner_prev = one(T)

    for k in 0:(D-1)
        inner_curr = dot(P_curr, P_curr) / N
        if abs(inner_curr) < eps(T)
            break
        end # Safety break

        α_k = dot(x .* P_curr, P_curr) / N / inner_curr
        β_k = inner_curr / inner_prev

        P_next = (x .- α_k) .* P_curr
        if k > 0
            P_next .-= β_k .* P_prev
        end

        # Recurrence on Coefficients
        for j in 0:k
            MonicCoeffs[k+2, j+2] += MonicCoeffs[k+1, j+1]         # x * P_k
            MonicCoeffs[k+2, j+1] -= α_k * MonicCoeffs[k+1, j+1]   # -α * P_k
        end
        if k > 0
            for j in 0:(k-1)
                MonicCoeffs[k+2, j+1] -= β_k * MonicCoeffs[k, j+1] # -β * P_{k-1}
            end
        end

        P_prev = copy(P_curr)
        P_curr = copy(P_next)
        inner_prev = inner_curr
    end
    return m, MonicCoeffs
end

function _v2_normalize_hankel(MonicCoeffs::AbstractMatrix{T}, m::AbstractVector{T}, D::Integer) where {T<:Real}
    OrthonormalCoeffs = zeros(T, D + 1, D + 1)
    for k in 0:D
        p = @view MonicCoeffs[k+1, 1:(k+1)]
        norm_sq = zero(T)
        for i in 0:k
            for j in 0:k
                norm_sq += p[i+1] * p[j+1] * m[i+j+1]
            end
        end
        norm_factor = sqrt(max(norm_sq, eps(T)))
        OrthonormalCoeffs[k+1, 1:(k+1)] = MonicCoeffs[k+1, 1:(k+1)] ./ norm_factor
    end
    return OrthonormalCoeffs
end

# ==============================================================================
# MAIN COMPARISON SCRIPT
# ==============================================================================

function run_comparison()
    # 1. Generate Synthetic Data
    # Using a Uniform distribution [0,1].
    # This leads to Legendre-like polynomials (shifted).
    # Moments remain bounded, making it a fair test for numerical stability.
    Random_Seed = 1234
    Data = rand(2000)

    println("================================================================")
    println("COMPARING aPCE IMPLEMENTATIONS")
    println("Data: 2000 samples, Uniform(0,1)")
    println("================================================================\n")

    degrees = [3, 7, 25]

    for D in degrees
        println("----------------------------------------------------------------")
        @printf("Checking Degree: %d\n", D)
        println("----------------------------------------------------------------")

        # Run Version 1 (Hankel)
        t1 = @elapsed begin
            C1 = aPCE_Hankel(Data, D, true)
        end
        for _ in 1:1000
            @timeit to "aPCE_Hankel_$D" C1 = aPCE_Hankel(Data, D, true)
        end

        # Run Version 2 (Stieltjes)
        t2 = @elapsed begin
            C2 = aPCE_Stieltjes(Data, D, true)
        end

        for _ in 1:1000
            @timeit to "aPCE_Stieltjes_$D" C2 = aPCE_Stieltjes(Data, D, true)
        end


        # Compare Coefficients
        # We compute the max absolute difference between the coefficient matrices
        diff_norm = norm(C1 - C2)
        max_diff = maximum(abs.(C1 - C2))

        @printf("V1 (Hankel) Time    : %.6f sec\n", t1)
        @printf("V2 (Stieltjes) Time : %.6f sec\n", t2)
        @printf("Max Coeff Diff      : %.4e\n", max_diff)

        # ----------------------------------------------------------------------
        # Verification: Orthogonality Check
        # ----------------------------------------------------------------------
        # We calculate the orthogonality error for the highest polynomial P_D
        # E[ P_D(x) * P_D(x) ] should be 1.0
        # E[ P_D(x) * P_{D-1}(x) ] should be 0.0

        function check_orth(Coeffs, data, d)
            # Evaluate polynomial degree d at data points
            P_d = zeros(length(data))
            for i in 1:length(data)
                val = 0.0
                for p in 0:d
                    val += Coeffs[d+1, p+1] * data[i]^p
                end
                P_d[i] = val
            end
            return P_d
        end

        # Evaluate last polynomial (Degree D) for both methods
        P_d_v1 = check_orth(C1, Data, D)
        P_d_v2 = check_orth(C2, Data, D)

        # Norm check (should be 1.0)
        norm_v1 = mean(P_d_v1 .^ 2)
        norm_v2 = mean(P_d_v2 .^ 2)

        @printf("\nOrthogonality Check (Ideal = 1.0):\n")
        @printf("V1 E[P_D^2] : %.6f  (Error: %.2e)\n", norm_v1, abs(norm_v1 - 1.0))
        @printf("V2 E[P_D^2] : %.6f  (Error: %.2e)\n", norm_v2, abs(norm_v2 - 1.0))

        if D == 21
            println("\n[ANALYSIS FOR DEGREE 21]")
            if abs(norm_v1 - 1.0) > 0.1
                println("-> Version 1 FAILED. The Hankel matrix is ill-conditioned.")
                println("   Standard floating point cannot solve the moment system M*c=rhs.")
            end
            if abs(norm_v2 - 1.0) < 0.1
                println("-> Version 2 SUCCEEDED (Relative to V1). Stieltjes is more stable.")
            end
        end
        println("\n")
    end
    display(to)
end

run_comparison()