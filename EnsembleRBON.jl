using LinearAlgebra
using Statistics
using Clustering
using Distances
using Base.Threads

# Set number of threads to 20
# if Threads.nthreads() != 20
#     @warn "Setting number of threads to 20. Current number of threads: $(Threads.nthreads())"
#     ENV["JULIA_NUM_THREADS"] = "20"
#     # Note: This will only take effect if set before Julia starts
#     # You may need to restart Julia for this to take effect
# end

include("src/RBON.jl")
include("src/RBON_ElasticNet.jl")

"""
    EnsembleRBON

A structure representing an ensemble of RBON models trained on different clusters of data.
"""
mutable struct EnsembleRBON
    n_models::Int
    models::Vector{RBON}
    cluster_centers::Matrix{Float64}
    cluster_sigmas::Vector{Float64}
    assignments::Vector{Int}
    similarity_threshold::Float64
    use_mlp::Bool
end

"""
    EnsembleRBON(n_models::Int, input_dim::Int, output_dim::Int, similarity_threshold::Float64=0.7, n_centers::Union{Int,Nothing}=nothing)

Create a new EnsembleRBON with specified number of models and dimensions.
If n_centers is provided, each RBON will use that many centers.
If n_centers is nothing, it will use min(50, input_dim ÷ 2) centers.
"""
function EnsembleRBON(n_models::Int, input_dim::Int, output_dim::Int, similarity_threshold::Float64=0.7, n_centers::Union{Int,Nothing}=nothing, use_mlp::Bool=false)
    # Use either specified number of centers or adaptive approach
    centers = isnothing(n_centers) ? min(50, input_dim ÷ 2) : n_centers
    models = [RBON(centers, input_dim, centers) for _ in 1:n_models]
    return EnsembleRBON(n_models, models, zeros(input_dim, n_models), zeros(n_models), Int[], similarity_threshold, use_mlp)
end

"""
    cosine_similarity(x::Vector{Float64}, y::Vector{Float64})

Calculate cosine similarity between two vectors.
"""
function cosine_similarity(x::Vector{Float64}, y::Vector{Float64})
    return dot(x, y) / (norm(x) * norm(y))
end

"""
    cosine_distance(x::Vector{Float64}, y::Vector{Float64})

Calculate cosine distance between two vectors (1 - cosine similarity).
"""
function cosine_distance(x::Vector{Float64}, y::Vector{Float64})
    return 1.0 - cosine_similarity(x, y)
end

"""
    find_closest_cluster(x::Vector{Float64}, centers::Matrix{Float64})

Find the closest cluster center for a given input vector using cosine similarity.
"""
function find_closest_cluster(x::Vector{Float64}, centers::Matrix{Float64})
    similarities = [cosine_similarity(x, centers[:, i]) for i in 1:size(centers, 2)]
    return argmax(similarities), maximum(similarities)
end

"""
    ensemble_rbon_fit!(ensemble::EnsembleRBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64}, α::Float64, λ::Float64)

Fit the ensemble of RBON models to the training data using cosine similarity for clustering.
"""
function ensemble_rbon_fit!(ensemble::EnsembleRBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64}, α::Float64, λ::Float64; use_mlp::Bool = false)
    # Perform k-means clustering on the input data using cosine distance
    data = Matrix{Float64}(U')
    
    # Initialize centers randomly
    n_samples = size(data, 1)
    center_indices = randperm(n_samples)[1:ensemble.n_models]
    centers = data[center_indices, :]
    
    # Initialize assignments
    assignments = zeros(Int, n_samples)
    old_assignments = ones(Int, n_samples)
    
    # K-means iteration with cosine distance
    while any(assignments .!= old_assignments)
        old_assignments = copy(assignments)
        
        # Assign points to nearest center using cosine similarity
        @threads for i in 1:n_samples
            similarities = [cosine_similarity(data[i, :], centers[j, :]) for j in 1:ensemble.n_models]
            assignments[i] = argmax(similarities)
        end
        
        # Update centers
        for j in 1:ensemble.n_models
            cluster_points = data[assignments .== j, :]
            if !isempty(cluster_points)
                # Compute mean direction (normalized sum of vectors)
                center_sum = sum(cluster_points, dims=1)[1, :]
                centers[j, :] = center_sum ./ norm(center_sum)
            end
        end
    end
    
    # Store cluster centers and assignments
    ensemble.cluster_centers = centers'
    ensemble.assignments = assignments
    
    # Calculate cluster sigmas using cosine distance
    @threads for i in 1:ensemble.n_models
        cluster_data = U[:, ensemble.assignments .== i]
        if size(cluster_data, 2) > 0
            ensemble.cluster_sigmas[i] = mean([cosine_distance(cluster_data[:, j], ensemble.cluster_centers[:, i]) for j in 1:size(cluster_data, 2)])
        end
    end
    
    # Train individual RBON models on each cluster in parallel
    for i in 1:ensemble.n_models
        cluster_indices = findall(ensemble.assignments .== i)
        if !isempty(cluster_indices)

            # Only proceed if enough samples
            if length(cluster_indices) < 2
                @warn "Skipping cluster $i: too few samples ($(length(cluster_indices)))"
                continue
            end

            U_cluster = U[:, cluster_indices]
            V_cluster = V[:, cluster_indices]
            n_cluster_samples = size(U_cluster, 2)

            # Cap h_num and l_num to be no greater than the number of samples
            h_centers = min(ensemble.models[i].h_num, n_cluster_samples)
            l_centers = min(ensemble.models[i].l_num, n_cluster_samples)

            # Redefine the RBON model with smaller network if needed
            ensemble.models[i] = RBON(h_centers, size(U, 1), l_centers)

            # Fit with clustering and training
            ensemble.models[i].h_centers, ensemble.models[i].h_sigmas, _ = find_best_kmeans(U_cluster, h_centers)
            ensemble.models[i].l_centers, ensemble.models[i].l_sigmas, _ = find_best_kmeans_1d(y, l_centers)
            
            # Fit the RBON model
            rbon_elasticnet_fit!(ensemble.models[i], U_cluster, V_cluster, y, α, λ, use_mlp = use_mlp)
        end
    end
end


"""
    ensemble_rbon_predict(ensemble::EnsembleRBON, U::Matrix{Float64}, y::Vector{Float64})

Make predictions using the ensemble of RBON models.
"""
function ensemble_rbon_predict(ensemble::EnsembleRBON, U::Matrix{Float64}, y::Matrix{Float64})
    n_samples = size(U, 2)
    predictions = zeros(size(y, 2), n_samples)

    for i in 1:n_samples
        x = U[:, i]
        cluster_idx, similarity = find_closest_cluster(x, ensemble.cluster_centers)
        
        if similarity < ensemble.similarity_threshold
            @warn "Sample $i is outside the domain of all clusters (similarity: $similarity). Using closest cluster $cluster_idx."
        end
        
        # Make prediction using the selected model
        pred = rbon_predict(ensemble.models[cluster_idx], reshape(x, :, 1), y)
        predictions[:, i] = pred[:, 1]
    end
    
    return predictions
end

"""
    ensemble_rbon_predict_topk(ensemble::EnsembleRBON, U::Matrix{Float64}, y::Matrix{Float64}, k::Int)

Make predictions by averaging top-k most similar RBON models, weighted by similarity.
"""
function ensemble_rbon_predict_topk(ensemble::EnsembleRBON, U::Matrix{Float64}, y::Matrix{Float64}, k::Int)
    n_samples = size(U, 2)
    predictions = zeros(size(y, 2), n_samples)

    for i in 1:n_samples
        x = U[:, i]
        similarities = [cosine_similarity(x, ensemble.cluster_centers[:, j]) for j in 1:ensemble.n_models]

        # Get top-k indices
        topk_idx = partialsortperm(similarities, rev=true, 1:k)
        topk_sim = similarities[topk_idx]

        # Normalize weights (softmax or sum-normalized)
        weights = topk_sim ./ sum(topk_sim)

        # Aggregate predictions
        pred_sum = zeros(size(y, 2))
        for (j, idx) in enumerate(topk_idx)
            pred = rbon_predict(ensemble.models[idx], reshape(x, :, 1), y)
            pred_sum .+= weights[j] .* vec(pred)
        end

        predictions[:, i] = pred_sum
    end

    return predictions
end



"""
    ensemble_rbon_evaluate(ensemble::EnsembleRBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Vector{Float64})

Evaluate the ensemble model on test data.
"""
function ensemble_rbon_evaluate(ensemble::EnsembleRBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64})
    predictions = ensemble_rbon_predict(ensemble, U, y)
    # Relative Error
    rel_errors = [norm(V[:, i] .- predictions[:, i]) / norm(V[:, i]) for i in 1:size(V, 2)]
    # Mean Squared Error
    mse = mean((V .- predictions).^2)
    mae = mean(abs.(V .- predictions))

    return mean(rel_errors), std(rel_errors), mse, mae, predictions
end 

"""
    ensemble_rbon_evaluate_topk(ensemble::EnsembleRBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64}, k::Int)

Evaluate the top-k weighted ensemble model on test data.
Returns mean relative error, std of relative error, MSE, MAE, and predictions.
"""
function ensemble_rbon_evaluate_topk(ensemble::EnsembleRBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64}, k::Int)
    predictions = ensemble_rbon_predict_topk(ensemble, U, y, k)

    # Relative Error
    rel_errors = [norm(V[:, i] .- predictions[:, i]) / norm(V[:, i]) for i in 1:size(V, 2)]

    # Mean Squared Error
    mse = mean((V .- predictions).^2)

    # Mean Absolute Error
    mae = mean(abs.(V .- predictions))

    return mean(rel_errors), std(rel_errors), mse, mae, predictions
end


"""
Try softmax
"""
function softmax(x::Vector{Float64})
    exps = exp.(x .- maximum(x))  # for numerical stability
    return exps ./ sum(exps)
end

function ensemble_rbon_predict_topk_softmax(ensemble::EnsembleRBON, U::Matrix{Float64}, y::Matrix{Float64}, k::Int)
    n_samples = size(U, 2)
    output_dim = size(y, 2)
    predictions = zeros(output_dim, n_samples)

    for i in 1:n_samples
        x = U[:, i]
        
        # Compute similarities to all cluster centers
        sims = [cosine_similarity(x, ensemble.cluster_centers[:, j]) for j in 1:ensemble.n_models]
        
        # Get top-k indices and similarities
        sorted_idxs = partialsortperm(sims, rev=true, 1:k)
        top_sims = sims[sorted_idxs]
        weights = softmax(top_sims)  # Apply softmax weighting

        # Weighted prediction
        pred = zeros(output_dim)
        for (j, w) in zip(sorted_idxs, weights)
            pred_j = rbon_predict(ensemble.models[j], reshape(x, :, 1), y)[:, 1]
            pred .+= w .* pred_j
        end
        predictions[:, i] = pred
    end

    return predictions
end

function ensemble_rbon_evaluate_topk_softmax(ensemble::EnsembleRBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64}, k::Int)
    predictions = ensemble_rbon_predict_topk_softmax(ensemble, U, y, k)
    
    rel_errors = [norm(V[:, i] .- predictions[:, i]) / norm(V[:, i]) for i in 1:size(V, 2)]
    mse = mean((V .- predictions).^2)
    mae = mean(abs.(V .- predictions))

    return mean(rel_errors), std(rel_errors), mse, mae, predictions
end

