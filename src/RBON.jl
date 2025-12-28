#cd(@__DIR__)
#using Pkg.activate(".")


using Clustering
using Plots
using ProgressMeter
using Flux


# Radial Basis Operator Network Class
mutable struct RBON
    h_num::Int
    l_num::Int
    h_centers::Matrix{Float64}
    l_centers::Matrix{Float64}
    h_sigmas::Vector{Float64}
    l_sigmas::Vector{Float64}
    weights::Vector{Float64}
    A::Union{Matrix{Float64}, Nothing}    # Affine transformation matrix
    b::Union{Vector{Float64}, Nothing}    # Affine bias vector
    mlp_model::Union{Chain, Nothing} # MLP if used
    
    # h denotes the higher network that takes function values as input
    # l denotes the lower network that takes the values where the output functions
    # are ultimately evaluated (domain location of output function values)
    function RBON(h_num_arg::Int, num_u_evals::Int, l_num_arg::Int)
        # Initialize fields using the constructor arguments
        l_centers_arg = zeros(l_num_arg,1)
        l_sigmas_arg = zeros(l_num_arg)
        h_centers_arg = zeros(num_u_evals, h_num_arg)
        h_sigmas_arg = zeros(h_num_arg)
        weights_arg = zeros(l_num_arg * h_num_arg)
        A_arg = nothing
        b_arg = nothing
        mlp_model_arg = nothing


        # random initialization of weights
        #Random.seed!(315)  # Set a seed for reproducibility
        #rbon.weights = randn(rbon.h_num*rbon.l_num)       
        new(h_num_arg, l_num_arg, h_centers_arg, l_centers_arg,  h_sigmas_arg, l_sigmas_arg, weights_arg, A_arg, b_arg, mlp_model_arg)
    end
end



# Fit the Radial Basis Network using projection operator training
function rbon_fit!(rbon::RBON, U::Matrix{Float64},  V::Matrix{Float64},  y::Matrix{Float64}; normalized::Bool = false)

    # === Cluster RBF centers ===
    h_result = kmeans(U, rbon.h_num)
    rbon.h_centers = h_result.centers
    rbon.h_sigmas = [sum(std(U[:, h_result.assignments .== i], mean=rbon.h_centers[:,i], dims=2)) for i in 1:rbon.h_num]
    rbon.h_sigmas = replace(rbon.h_sigmas, NaN => 1.0)

    rbon.l_centers, l_assignments = kmeans_1d(y, rbon.l_num)
    rbon.l_sigmas = [sum(std(y[:, l_assignments .== i], mean=rbon.l_centers[:,i], dims=2)) for i in 1:rbon.l_num]
    rbon.l_sigmas = replace(rbon.l_sigmas, NaN => 1.0)

    # === Prepare training data ===
    num_input_funcs = size(U, 2)
    output_dim = size(y, 2)
    d = rbon.h_num * rbon.l_num
    all_weights = zeros(d, output_dim)

    # Compute RBF activations
    H = reduce(hcat, [radial_basis_function(U, rbon.h_centers[:, i], rbon.h_sigmas[i]) for i in 1:rbon.h_num])'
    L = reduce(hcat, [radial_basis_function(y, rbon.l_centers[:, i], rbon.l_sigmas[i]) for i in 1:rbon.l_num])'

    p = Progress(output_dim, 1, "Fitting RBON weights", 0)  # initialize progress bar

    Threads.@threads for j in 1:output_dim
        # Compute feature matrix Φ_j for all inputs U at fixed y_j
        Φ = reduce(hcat, [(H[:, k] * L[:, j]')[:] for k in 1:num_input_funcs])  # shape: d x n

        if normalized
            Φ .= Φ ./ sum(Φ; dims=1)  # normalize each column
        end

        # Solve least squares: Φᵗ * w = V[j, :]
        weight_j = pinv(Φ)' * V[j, :]
        all_weights[:, j] = weight_j

        next!(p)  # update progress bar
    end

    # Average weights across all output points
    rbon.weights = mean(all_weights, dims=2)[:, 1]

    # === Compute predictions for affine layer calibration ===
    predictions = zeros(output_dim, num_input_funcs)

    for k in 1:num_input_funcs
        for j in 1:output_dim
            Phi_vec = vec(H[:, k] * L[:, j]')
            if normalized
                Phi_vec ./= sum(Phi_vec)
            end
            predictions[j, k] = dot(Phi_vec, rbon.weights)
        end
    end

    # === Affine transformation layer (A and b) ===
    pred_aug = [predictions; ones(1, size(predictions, 2))]
    AB = V * pinv(pred_aug)
    rbon.A = AB[:, 1:output_dim]
    rbon.b = AB[:, output_dim + 1]
end




















# Predict with the Radial Basis Operator Network
function rbon_predict(rbon::RBON, U::Matrix{Float64}, y::Matrix{Float64}; 
    normalized::Bool = false)#, linear_output::Bool = true)
    num_input_funcs = size(U, 2)
    num_target_pts = size(y, 2)

    # Initialize an empty matrix to store the results
    predictions = zeros(Float64, num_target_pts, num_input_funcs)
    preds = zeros(Float64, num_target_pts, num_input_funcs)

    # Compute radial basis function activations
    H = reduce(hcat, [radial_basis_function(U, rbon.h_centers[:, i], rbon.h_sigmas[i]) for i in 1:rbon.h_num])'
    L = reduce(hcat, [radial_basis_function(y, rbon.l_centers[:, i], rbon.l_sigmas[i]) for i in 1:rbon.l_num])'

    Threads.@threads for j in 1:num_input_funcs
        for k in 1:num_target_pts
            Phi = H[:, j] * L[:, k]'  # h_num x l_num matrix
            Phi_flat = vec(Phi)       # flatten to vector

            weighted_sum = dot(Phi_flat, rbon.weights)

            if normalized
                predictions[k, j] = weighted_sum / sum(Phi_flat)
            else
                predictions[k, j] = weighted_sum
            end
        end
    end

    predictions = rbon.A * predictions .+ rbon.b  # Apply affine transformation

    # If MLP model is used, apply it to the predictions
    if !isnothing(rbon.mlp_model)
        X = predictions
        preds = rbon.mlp_model(X)
    else
        preds = predictions  # If no tree or MLP, use affine transformation
    end

    return preds
end


# Example usage:
# Create a Radial Basis Network with 50 nodes for the higher network, 100 nodes for lower network
# knowing the number of evals for training functions used for initialization purposes, can be altered as needed
#rbon = RBON(50, size(U,1), 100)