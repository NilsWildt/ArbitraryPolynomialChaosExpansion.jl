using Random

@kernel function outer_product_kernel!(output, Ψ, expansion_coefficients)
    i, j = @index(Global, NTuple)
    for k in 1:size(output, 1)
        @inbounds output[k, j] += Ψ[i, k] * expansion_coefficients[i, j]
    end
end

# Creating a wrapper kernel for launching with error checks
function outer_product!(output, Ψ, expansion_coefficients)
    # if size(Ψ)[1] != size(expansion_coefficients)[1]
    #     println("Matrix size mismatch!")
    #     return nothing
    # end
    backend = KernelAbstractions.get_backend(Ψ)
    kernel! = outer_product_kernel!(backend)
    return kernel!(output, Ψ, expansion_coefficients, ndrange = size(expansion_coefficients))
end

@inline function outer_product_kernel(Ψ::AbstractArray{T}, α) where {T <: Real}
    (i_dim, k_dim) = size(Ψ)
    (i_dim_exp, j_dim) = size(α)
    backend = get_backend(Ψ) # or KernelAbstractions.CUDA() for GPU
    KernelAbstractions.synchronize(backend)
    out = KernelAbstractions.zeros(backend, T, k_dim, j_dim)
    outer_product!(out, Ψ, α)
    return Array{T}(out)
end


function removeNaN(x)
    return isnan(x) ? Float32(0.0) : x
end

"""
# Returns

- `X_train`: The training set features.
- `X_test`: The testing set features.

The function randomly shuffles the dataset and splits it according to the specified training proportion (`at`).
"""
function partitionTrainTest(data; at = 0.7, rng = Xoshiro())
    num_samples = size(data, 1)
    shuffled_indices = shuffle(rng, 1:num_samples)
    split_index = floor(Int, at * num_samples)

    train_indices = view(shuffled_indices, 1:split_index)
    test_indices = view(shuffled_indices, (split_index + 1):num_samples)

    X_train = data[train_indices]
    X_test = data[test_indices]

    return X_train, X_test
end


# Macro for checking arguments
macro check_args(K, param, cond, desc = string(cond))
    return quote
        if !($(esc(cond)))
            throw(
                ArgumentError(
                    string(
                        $(string(K)),
                        ": ",
                        $(string(param)),
                        " = ",
                        $(esc(param)),
                        " does not ",
                        "satisfy the constraint ",
                        $(string(desc)),
                        ".",
                    ),
                ),
            )
        end
    end
end
