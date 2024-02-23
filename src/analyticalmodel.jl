using LinearAlgebra

"""
    AnalyticalModel

Module contains the class for the analytical model, composed of a non-linear equation, obtained from:

- Oladyshkin, S., Mohammadi, F., Kroeker I. and Nowak, W. Bayesian active learning for Gaussian process emulator using
    information theory. Entropy, X(X), X, 2020,
- Oladyshkin, S. and Nowak, W. The Connection between Bayesian Inference and Information Theory for Model Selection,
    Information Gain and Experimental Design. Entropy, 21(11), 1081, 2019,

For function details and reference information, see:
https://doi.org/10.3390/e21111081
"""
mutable struct AnalyticalModel
    func::String
    loc::Array{Float64,2}
    observations::Array{Float64,2}
    n_obs::Int
    n_params::Int
    var::Float64

    AnalyticalModel(func::String="non-linear", loc::Array{Float64,2}=Array{Float64,2}(undef, 0, 1), observations::Array{Float64,2}=Array{Float64,2}(undef, 0, 0)) = new(func, loc, observations)
end

function check_input(model::AnalyticalModel)
    # Placeholder for input checking logic
end

function nonlinear_model(params::Array{Float64,2}, loc::Array{Float64,1})
    if ndims(params) == 1
        params = reshape(params, 1, size(params, 1))
    end
    if size(params, 2) == 1
        params = hcat(params, params)
    end

    n_params = size(params, 2)
    param_sets = size(params, 1)

    term1 = (params[:, 1] .^ 2 .+ params[:, 2] .- 1) .^ 2
    term2 = params[:, 1] .^ 2
    term3 = 0.1 .* params[:, 1] .* exp.(params[:, 2])
    
    term5 = zeros(param_sets)
    if n_params > 2
        for i in 3:n_params
            term5 .+= params[:, i] .^ 3 / (i + 1)
        end
    end

    const_per_set = term1 .+ term2 .+ term3 .+ term5 .+ 1
    term4 = zeros(param_sets, length(loc))
    for i in 1:param_sets
        term4[i, :] = -2 * params[i, 1] * sqrt.(0.5 * loc)
    end

    output = term4 .+ repeat(const_per_set, 1, length(loc))
    return output
end

function evaluate_model(model::AnalyticalModel, params::Array{Float64,2})
    output = Array{Float64,2}(undef, 0, 0)
    if model.func == "non_linear"
        output = nonlinear_model(params, model.loc)
    end
    return output
end