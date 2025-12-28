using Clustering
using GLMNet, Random, Statistics
using ProgressMeter
using DecisionTree
using Flux
using Flux: DataLoader


# Fit the Radial Basis Network using <strike>projection</strike> shrinkage operator training
function rbon_elasticnet_fit!(
    rbon::RBON,
    U::Matrix{Float64},
    V::Matrix{Float64},
    y::Matrix{Float64},
    α::Float64,
    λ::Float64;
    normalized::Bool = false,
    use_mlp::Bool = false
)

    num_input_funcs = size(U, 2)
    output_dim = size(y, 2)
    d = rbon.l_num * rbon.h_num
    all_weights = Matrix{Float64}(undef, d, num_input_funcs)

    # Compute RBF activations
    H = reduce(hcat, [radial_basis_function(U, rbon.h_centers[:, i], rbon.h_sigmas[i]) for i in 1:rbon.h_num])'
    L = reduce(hcat, [radial_basis_function(y, rbon.l_centers[:, i], rbon.l_sigmas[i]) for i in 1:rbon.l_num])'

    p = Progress(num_input_funcs, 1, "Fitting ElasticNet weights")
    @sync Threads.@threads for k in 1:num_input_funcs
        Φ = reduce(hcat, [(H[:, k] * L[:, j]')[:] for j in 1:output_dim])
        if normalized
            Φ .= Φ ./ sum(Φ; dims=1)
        end
        model = glmnet(Φ', V[:, k], alpha=α, lambda=[λ])
        all_weights[:, k] = convert(Matrix, model.betas)
        next!(p)
    end

    rbon.weights = mean(all_weights, dims=2)[:, 1]

    # Generate RBON features (predictions before affine or tree)
    predictions = zeros(output_dim, num_input_funcs)
    for k in 1:num_input_funcs
        for j in 1:output_dim
            phi_vec = vec(H[:, k] * L[:, j]')
            if normalized
                phi_vec ./= sum(phi_vec)
            end
            predictions[j, k] = dot(phi_vec, rbon.weights)
        end
    end

    # === Use affine transformation ===
    pred_aug = [predictions; ones(1, size(predictions, 2))]
    AB = V * pinv(pred_aug)
    rbon.A = AB[:, 1:output_dim]
    rbon.b = AB[:, output_dim + 1]
    predictions = rbon.A * predictions .+ rbon.b

    if use_mlp
        # Prepare training data
        X = Float32.(predictions)  # size: (num_features, num_samples)
        Y = Float32.(V)            # size: (output_dim, num_samples)

        input_dim = size(X, 1)
        output_dim = size(Y, 1)

        # Define a single MLP model
        rbon.mlp_model = Chain(
            Dense(input_dim, input_dim+100, gelu_tanh),
            Dense(input_dim+100, output_dim)
        )

        # Define loss and optimiser
        loss(model, x, y) = Flux.Losses.mse(model(x), y)
        opt = Flux.setup(Flux.Adam(), rbon.mlp_model)

        # Transpose to column-major for Flux (features in columns)
        dataset = [(X, Y)]
        #dataloader = DataLoader((X, Y), batchsize=32, shuffle=true)

        # Train model
        #Flux.train!(loss, rbon.mlp_model, dataset, opt)
        for epoch in 1:1000 
            Flux.train!(loss, rbon.mlp_model, dataset, opt)
            if epoch % 100 == 0
                println("Epoch $epoch - MSE: ", loss(rbon.mlp_model, X, Y))
            end
        end  
    else
        rbon.mlp_model = nothing
    end
end

#=
rbon_elasticnet_fit!(rbon, U_train, V_train, y, 0.5, 0.001)


# Cross-validation function to optimize α, λ
function rbon_elasticnet_cv!(rbon::RBON, U::Matrix{Float64}, V::Matrix{Float64}, y::Matrix{Float64};
                             alphas::Vector{Float64} = range(0.1, stop=1.0, length=5),
                             lambdas::Vector{Float64} = range(0.001, stop=1.0, length=50),
                             n_folds::Int = 5, seed::Int = 1234)

    # Set seed for reproducibility
    Random.seed!(seed)

    # Number of data points
    n_samples = size(U, 2)

    # Split data into k folds
    indices = collect(1:n_samples)
    Random.shuffle!(indices)
    folds = [indices[i:step:end] for i in 1:n_folds for step in [div(n_samples, n_folds)]]

    best_alpha, best_lambda = 0.0, 0.0
    best_error = Inf

    # Iterate over all combinations of α, λ_1, λ_2
    for α in alphas
        for λ in lambdas
            # Initialize a list to store the errors for each fold
            fold_errors = Float64[]

            # Cross-validation: loop over each fold
            for i in 1:n_folds
                # Split the data into training and validation sets
                val_idx = folds[i]
                train_idx = setdiff(1:n_samples, val_idx)

                # Training data
                U_train = U[:, train_idx]
                V_train = V[:, train_idx]

                # Validation data
                U_val = U[:, val_idx]
                V_val = V[:, val_idx]


                # Fit the model on training data
                rbon_elasticnet_fit!(rbon, U_train, V_train, y, α, λ)

                # Make predictions on the validation data
                val_predictions = zeros(size(V_val))
                val_predictions = rbon_predict(rbon, U_val, y)


                # Calculate the mean squared error for this fold
                error = mean((V_val - val_predictions).^2)
                push!(fold_errors, error)
            end

            # Average the validation error across all folds
            avg_error = mean(fold_errors)
            println("Current λ:", λ,  " current error: ", avg_error)
            # If this combination is better, update the best parameters
            if avg_error < best_error
                best_error = avg_error
                best_alpha, best_lambda = α, λ
            end
        end
    end

    println("Best α: ", best_alpha)
    println("Best λ: ", best_lambda)
    println("Best CV error: ", best_error)

    # Train the model one last time with the best parameters on the full dataset
    rbon_elasticnet_fit!(rbon, U, V, y, best_alpha, best_lambda)

    return best_alpha, best_lambda, best_error
end


U_train, U_test, V_train, V_test = train_test_split(U, V, 0.8)

# Perform cross-validation to find the best alpha, lambda_1, and lambda_2
rbon = RBON(21, size(U_train, 1), 21)

# Find best k-means clustering for higher network
rbon.h_centers, rbon.h_sigmas, h_assignments = find_best_kmeans(U_train, rbon.h_num)
#rbon.h_centers, rbon.h_sigmas, h_assignments = find_worst_kmeans(U_train, rbon.h_num)

# Find best k-means clustering for lower network
rbon.l_centers, rbon.l_sigmas, l_assignments = find_best_kmeans_1d(y, rbon.l_num)
#rbon.l_centers, rbon.l_sigmas, l_assignments = find_worst_kmeans_1d(y, rbon.l_num)



best_alpha, best_lambda, best_error = rbon_elasticnet_cv!(
    rbon, U_train, V_train, y;
    alphas = collect(range(0.0, stop=1.0, length=5)),
    lambdas = collect(range(0.001, stop=1.0, length=5)), n_folds = 2)



α=0.25
    
λ=1.0

rbon_elasticnet_fit!(rbon, U_train, V_train, y, α, λ)

# L2 relative test error, in distribution
predictions = rbon_predict(rbon, U_test, y)
rel_errors = [norm(V_test[:, i] .- predictions[:, i]) / norm(V_test[:, i]) for i in 1:size(V_test, 2)]

average_error = mean(rel_errors)

minimum(rel_errors)
maximum(rel_errors)

min_rel_norm = Inf
best_i, best_j = 2, 2 # set starting values

for i in 4:15
    for j in 2:15
        Random.seed!(1)
        # Create and fit the RBON
        temp_rbon = RBON(i, size(U_train, 1), j)
        # Find best k-means clustering for higher network
        rbon.h_centers, rbon.h_sigmas, h_assignments = find_best_kmeans(U_train, temp_rbon.h_num)

        # Find best k-means clustering for lower network
        rbon.l_centers, rbon.l_sigmas, l_assignments = find_best_kmeans_1d(y, temp_rbon.l_num)

        best_alpha, best_lambda, best_error = rbon_elasticnet_cv!(temp_rbon, U_train, V_train, y; alphas = collect(range(0.0, stop=1.0, length=5)), lambdas = collect(range(0.001, stop=1.0, length=5)))

        rbon_elasticnet_fit!(temp_rbon, U_train, V_train, y, best_alpha, best_lambda)
        
        # Predict
        predictions = rbon_predict(temp_rbon, U_test, y)
        
        # Compute the relative norm
        rel_norm = norm(V_test .- predictions) / norm(V_test)
        
        # Check if this is the best so far
        if rel_norm < min_rel_norm
            min_rel_norm = rel_norm
            best_i, best_j = i, j
            printstyled("new min val error = $min_rel_norm, ", color=:blue)
            printstyled("new best i = $best_i, ", color=:green)
            printstyled("new best j = $best_j, ", color=:red)
        end
        print("Network widths tested this iteration: $i, $j\n ")

    end
end
=#