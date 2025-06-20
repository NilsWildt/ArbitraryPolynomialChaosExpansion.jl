### A Pluto.jl notebook ###
# v0.20.10
ccall(:jl_exit_on_sigint, Nothing, (Cint,), true)

using Markdown
using InteractiveUtils

# ╔═╡ 14f0b662-5cbe-436a-874e-731a02685a8a
begin
    using TypeUtils
    using DispatchDoctor
    using Statistics
    using ConcreteStructs
    using StableRNGs
    using InteractiveErrors
    using PrettyChairmarks
    using ChainRulesCore
    using Test
    using TestItems
    using DifferentiationInterface
    using DifferentiationInterfaceTest
    import ReverseDiff
    import ForwardDiff
    import Mooncake
    using LinearAlgebra
    import Zygote
    import Enzyme
    using Enzyme.API: @import_rrule

end

# ╔═╡ 1f9285fc-4dd4-11f0-288b-3ddd7bbf925f

@stable @inbounds function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S, center_data::Val{true}) where {T <: Real, S <: Integer}
    d = Degree #Degree of polynomial expansion
    dd = d #Degree of polynomial for roots definition
    NumberOfDataPoints = length(Data)
    MeanOfData = mean(Data)
    Data_scaled = Data ./ MeanOfData  # scale data by division (not subtraction)

    # Compute moments using scaled data
    m = zeros(T, 2 * dd + 2)
    for i in 0:(2 * dd + 1)
        m[i + 1] = sum(Data_scaled .^ i) / NumberOfDataPoints
    end

    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1)

    for degree in 0:dd
        Hankel = @views OrthogonalBasis[1:(degree + 1), 1:(degree + 1)]
        Vc = zeros(T, degree + 1)

        for i in 0:(degree - 1)
            for j in 0:degree
                Hankel[i + 1, j + 1] = m[i + j + 1]
            end
            max_val = maximum(abs.(@views Hankel[i + 1, :]))
            if max_val > zero(T)
                Hankel[i + 1, :] = @views Hankel[i + 1, :] / max_val
            end
        end
        for j in 0:(degree - 1)
            Hankel[degree + 1, j + 1] = zero(T)
        end
        Hankel[degree + 1, degree + 1] = one(T)
        max_val = maximum(abs.(@views Hankel[degree + 1, :]))
        if max_val > zero(T)
            Hankel[degree + 1, :] = @views Hankel[degree + 1, :] / max_val
        end

        # Loop for Vc
        for i in 0:(degree - 1)
            Vc[i + 1] = zero(T)
        end
        Vc[degree + 1] = one(T)

        try
            OrthogonalBasis[degree + 1, 1:(degree + 1)] .= Hankel \ Vc
        catch
            @debug "Linear solve failed for polynomial basis of degree $degree. Trying pseudo-inverse."
            local basis_row
            try
                basis_row = pinv(Hankel) * Vc
            catch
                basis_row = nothing
            end

            if basis_row === nothing || any(!isfinite, basis_row)
                @debug "Pseudo-inverse failed or resulted in non-finite values for degree $degree. Falling back to standard monomial basis."
                fill!(@view(OrthogonalBasis[degree + 1, 1:degree]), zero(T))
                OrthogonalBasis[degree + 1, degree + 1] = one(T)
            else
                OrthogonalBasis[degree + 1, 1:(degree + 1)] .= basis_row
            end
        end

        # Check computational error
        if 100 * abs(sum(abs.(Hankel * OrthogonalBasis[degree + 1, 1:(degree + 1)])) - sum(abs.(Vc))) > 0.5
            deviation = 100 * abs(sum(abs.(Hankel * OrthogonalBasis[degree + 1, 1:(degree + 1)])) - sum(abs.(Vc))) / sum(abs.(Vc))
            # @warn "Computational error of the linear solver is too high: $(round(deviation; digits = 3))% for polynomial basis of degree $degree"
        end

        #Normalization of polynomial coefficients using scaled data
        P_norm = zero(T)
        for i in 1:NumberOfDataPoints
            Poly = zero(T)
            for k in 0:degree
                Poly += @views OrthogonalBasis[degree + 1, k + 1] * Data_scaled[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end
        for k in 0:degree
            OrthonormalBasis[degree + 1, k + 1] = @views OrthogonalBasis[degree + 1, k + 1] / sqrt(P_norm)
        end
    end

    # Backward transformation to data space (matching MATLAB implementation)
    # First scale data back
    # Then transform basis column-wise like in MATLAB
    for k in 1:lastindex(OrthonormalBasis, 2)
        OrthonormalBasis[:, k] = OrthonormalBasis[:, k] ./ (MeanOfData^(k - 1))
    end

    return OrthonormalBasis
end

# ╔═╡ 616701d9-192b-45ce-bc61-23d545479a0f
@stable @inbounds function aPCE_OrthonormalBasis(Data::AbstractArray{T}, Degree::S, center_data::Val{false}) where {T <: Real, S <: Integer}
    d = Degree #Degree of polynomial expansion
    dd = d #Degree of polynomial for roots definition
    NumberOfDataPoints = length(Data)

    # Compute raw moments exactly as in MATLAB
    m = zeros(T, 2 * dd + 2)
    for i in 0:(2 * dd + 1)
        m[i + 1] = sum(Data .^ i) / NumberOfDataPoints
    end

    OrthonormalBasis = zeros(T, dd + 1, dd + 1)
    OrthogonalBasis = zeros(T, dd + 1, dd + 1)

    for degree in 0:dd
        Hankel = @views OrthogonalBasis[1:(degree + 1), 1:(degree + 1)]
        Vc = zeros(T, degree + 1)

        for i in 0:(degree - 1)
            for j in 0:degree
                Hankel[i + 1, j + 1] = m[i + j + 1]
            end
            max_val = maximum(abs.(@views Hankel[i + 1, :]))
            if max_val > zero(T)
                Hankel[i + 1, :] = @views Hankel[i + 1, :] / max_val
            end
        end
        for j in 0:(degree - 1)
            Hankel[degree + 1, j + 1] = zero(T)
        end
        Hankel[degree + 1, degree + 1] = one(T)
        max_val = maximum(abs.(@views Hankel[degree + 1, :]))
        if max_val > zero(T)
            Hankel[degree + 1, :] = @views Hankel[degree + 1, :] / max_val
        end

        # Loop for Vc
        for i in 0:(degree - 1)
            Vc[i + 1] = zero(T)
        end
        Vc[degree + 1] = one(T)

        try
            OrthogonalBasis[degree + 1, 1:(degree + 1)] .= Hankel \ Vc
        catch
            OrthogonalBasis[degree + 1, 1:(degree + 1)] .= pinv(Hankel) * Vc
            # @warn "Used pinv for polynomial basis of degree $degree"
        end

        # Check computational error
        if 100 * abs(sum(abs.(Hankel * OrthogonalBasis[degree + 1, 1:(degree + 1)])) - sum(abs.(Vc))) > 0.5
            deviation = 100 * abs(sum(abs.(Hankel * OrthogonalBasis[degree + 1, 1:(degree + 1)])) - sum(abs.(Vc))) / sum(abs.(Vc))
            # @warn "Computational error of the linear solver is too high: $(round(deviation; digits = 3))% for polynomial basis of degree $degree"
        end

        #Normalization of polynomial coefficients using original data
        P_norm = zero(T)
        for i in 1:NumberOfDataPoints
            Poly = zero(T)
            for k in 0:degree
                Poly += @views OrthogonalBasis[degree + 1, k + 1] * Data[i]^k
            end
            P_norm += Poly^2 / NumberOfDataPoints
        end
        for k in 0:degree
            OrthonormalBasis[degree + 1, k + 1] = @views OrthogonalBasis[degree + 1, k + 1] / sqrt(P_norm)
        end
    end

    return OrthonormalBasis
end


# ╔═╡ 41050f3a-9538-4496-bb94-d73d41ce6181
@stable @inline function compute_moments!(m::AbstractArray{T}, Data::AbstractArray{S}, NumberOfDataPoints::Integer, dd::Integer) where {T <: Real, S <: Real}
    current_power = Vector{T}(undef, length(Data))  # Pre-allocate with correct type
    fill!(current_power, one(T))  # Initialize with ones of correct type
    for l in 0:(2 * dd + 1)
        m[l + 1] = sum(current_power) / NumberOfDataPoints
        current_power .*= Data  # Increment the power of Data
    end
    return nothing
end

# ╔═╡ d74f48c9-a8b4-4ccc-af7d-122c1948a6f0
@stable function aPCE_FullBasis(Data, Degree)
    T = eltype(Data)
    d = Degree #Degree of polynomial expansion
    dd = d #Degree of polynomial for roots definition
    # FullBasis = [i >= j ? one(T) : zero(T) for i in 1:dd+1, j in 1:dd+1]
    # # Do this FullBasis = [i >= j ? one(T) : zero(T) for i in 1:dd+1, j in 1:dd+1] as pre allocated for loop to be type stable
    FullBasis = Matrix{T}(undef, Degree + 1, Degree + 1)

    # Fill the matrix using a type-stable for loop
    for i in 1:(Degree + 1)
        for j in 1:(Degree + 1)
            if i >= j
                FullBasis[i, j] = one(T)
            else
                FullBasis[i, j] = zero(T)
            end
        end
    end
    return FullBasis
end


# ╔═╡ 014fc1e1-2904-42cd-8767-4e5a4d872296
begin
    function create_basis(x, degree; center_data = true)
        return create_basis(x, degree, Val(true); center_data = center_data)
    end


    function create_basis(x, degree, is_orthonormal::Val{true}; center_data = true)
        input_dimensions = size(x, 2)
        OrthonormalBasis = Array{eltype(x), 3}(undef, degree + 1, degree + 1, input_dimensions)
        for i in 1:input_dimensions
            OrthonormalBasis[:, :, i] .= aPCE_OrthonormalBasis(view(x, :, i), degree, Val(center_data))
        end # x[:,i]
        return OrthonormalBasis
    end


    function create_basis(x, degree, is_orthonormal::Val{false}; center_data = true)
        input_dimensions = size(x, 2)
        OrthonormalBasis = Array{eltype(x), 3}(undef, degree + 1, degree + 1, input_dimensions)
        for i in 1:input_dimensions
            OrthonormalBasis[:, :, i] .= aPCE_FullBasis(view(x, :, i), degree)
        end
        return OrthonormalBasis
    end

    function create_basis!(OrthonormalBasis, x, degree; center_data = false)
        # @ignore_derivatives begin
        input_dimensions = size(x, 2)
        # OrthonormalBasis = eltype(x).(OrthonormalBasis)
        if eltype(OrthonormalBasis) != eltype(x)
            OrthonormalBasis = eltype(x).(OrthonormalBasis)
        end
        # OrthonormalBasis = zeros(eltype(x), degree + 1, degree + 1, input_dimensions)
        for i in 1:input_dimensions
            OrthonormalBasis[:, :, i] .= aPCE_OrthonormalBasis(view(x, :, i), degree, Val(center_data))
        end
        # return OrthonormalBasis
        # end
        return
    end
end

# ╔═╡ bcfefdb0-2f0d-4b0c-992f-6d80b8536b30
# function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray, d_expansion::Int, is_orthonormal::Val{B}; center_data=true) where {B}
#     basis = create_basis(x, d_expansion, is_orthonormal; center_data=center_data)
#     function create_basis_pullback(Δbasis)
#         function basis_wrapper(x_vec)
#             x_reshaped = reshape(x_vec, size(x))
#             basis_val = create_basis(x_reshaped, d_expansion, is_orthonormal; center_data=center_data)
#             return sum(basis_val .* unthunk(Δbasis))
#         end
#         grad = ForwardDiff.gradient(basis_wrapper, vec(x))
#         grad_reshaped = reshape(grad, size(x))
#         return (NoTangent(), grad_reshaped, NoTangent(), NoTangent())
#     end
#     return basis, create_basis_pullback
# end

function ChainRulesCore.rrule(::typeof(create_basis), x::AbstractArray{T}, d_expansion::Int; center_data=false) where {T}
    # Forward pass
    basis = create_basis(x, d_expansion; center_data=center_data)
    
    # Define the pullback
    function create_basis_pullback(Δbasis)
        # Use ForwardDiff to compute the gradient
        function basis_wrapper(x_vec)
            x_reshaped = reshape(x_vec, size(x))
            basis = create_basis(x_reshaped, d_expansion; center_data=center_data)
            # Return a scalar value for gradient computation
            return sum(basis .* Δbasis)
        end
        
        # Compute gradient using ForwardDiff
        grad = ForwardDiff.gradient(basis_wrapper, vec(x))
        
        # Reshape gradient to match input shape
        grad_reshaped = reshape(grad, size(x))
        
        # Return the gradient with respect to x
        return (NoTangent(), grad_reshaped, NoTangent())
    end
    
    return basis, create_basis_pullback
end



# Enzyme.@import_rrule(typeof(create_basis), AbstractArray, Integer, Val)

ReverseDiff.@grad_from_chainrules create_basis(
    x::ReverseDiff.TrackedArray, d::Integer, is_orthonormal::Val
);

# ╔═╡ 390a894b-2689-43d1-adf6-a84b95698b83
let
    N = 100
    d_in = 1
    xs = rand(StableRNG(1), N, d_in)
    x = xs .^ 3 .+   rand(StableRNG(123), size(xs))
    d_out = 2

    f_true_centered(x_in) = sum(create_basis(x_in, d_out, Val(true); center_data = true))
    f_true(x_in) = sum(create_basis(x_in, d_out, Val(true); center_data = false))
    f_false(x_in) = sum(create_basis(x_in, d_out, Val(false)))

    backends = [AutoZygote(), AutoForwardDiff(), AutoReverseDiff(;compile=true),AutoMooncake(;config=nothing),AutoEnzyme()]

    ∇f_true = x ->ForwardDiff.gradient(f_true, x)
    ∇f_true_centered =x -> ForwardDiff.gradient(f_true_centered, x)
    ∇f_false = x -> ForwardDiff.gradient(f_false, x)

    @info "" (@bs f_true(xs)) f_true(x) ∇f_true(x)
    @info "" (@bs f_true(xs)) f_false(x) ∇f_false(x)
    @info "" (@bs f_true(xs)) f_true_centered(x) ∇f_true_centered(x)


    scenarios = [
        Scenario{:gradient, :out}(f_true, x;res1=∇f_true(x)), Scenario{:gradient, :out}(f_false, x;res1=∇f_false(x)),Scenario{:gradient, :out}(f_true_centered, x;res1=∇f_true_centered(x)),
    ]



    println("Testing AD for create_basis")
    test_differentiation(
        backends,
        scenarios;
        logging = true,
        allocations = :none,
        benchmark = :none,
        correctness = true,
        type_stability = :none,
        detailed = true,
        atol=1e-3,
        rtol=1e-3,
        count_calls=true,
        scenario_intact=true
    )
end
