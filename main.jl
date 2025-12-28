# Repository-ready version of main.jl
# Update paths to match repository structure: data/ and src/

using MAT
using StatsBase
using Printf
using Base.Threads
include("src/utils.jl")
include("EnsembleRBON.jl")

# Report number of threads
println("Number of threads: $(Threads.nthreads())")
println()

# Load the EIT data
reader1 = matread("data/EIT_4/EIT_FineSamples1.mat")
reader2 = matread("data/EIT_4/EIT_FineSamples2.mat")
reader1or2 = matread("data/EIT_4/EIT_FineSamples1or2.mat")
reader = matread("data/EIT_4/FineGridPoints.mat")

readerc = matread("data/Ensemble/EIT_2_CoarseSamples.mat")
readerc2 = matread("data/Ensemble/CoarseGridPoints.mat")

# Read the fields
U = readerc["phiCoarseSave"] # matrix of initial condition functions (input)
V = readerc["sigCoarseSave"] # matrix of solutions
y_coarse = readerc2["p"] # domain location of solutions

U1 = reader1["phiFineSave"] # matrix of initial condition functions (input)
V1 = reader1["sigFineSave"] # matrix of solutions
U2 = reader2["phiFineSave"] # matrix of initial condition functions (input)
V2 = reader2["sigFineSave"] # matrix of solutions
U1or2 = reader1or2["phiFineSave"] # matrix of initial condition functions (input)
V1or2 = reader1or2["sigFineSave"] # matrix of solutions
y = reader["p1"] # domain location of solutions

using Random
Random.seed!(222)

# Prepare the three datasets
datasets = [
    ("Anomaly1", U1, V1, y),
    ("Anomaly1or2", U1or2, V1or2, y),
    ("Anomaly2", U2, V2, y)
]

# Number of models to test
n_models_list = [1, 5, 10, 15]

# Regularization parameters
α = 0.25
λ = 1.0

# Results storage
results = []

println("="^80)
println("Ensemble RBON Training Time Comparison")
println("="^80)
println()

# Iterate over datasets
for (dataset_name, U_data, V_data, y_data) in datasets
    println("Dataset: $dataset_name")
    println("-"^80)
    
    # Split the data
    Random.seed!(222)
    U_train, U_test, V_train, V_test = train_test_split(U_data, V_data, 0.8)
    
    println("Training samples: $(size(U_train, 2)), Test samples: $(size(U_test, 2))")
    println()
    
    # Iterate over number of models
    for n_models in n_models_list
        println("  Testing with n_models = $n_models...")
        
        # Create ensemble
        ensemble = EnsembleRBON(n_models, size(U_train, 1), size(V_train, 1), 0.8, 21, false)
        
        # Time the fitting
        fit_time = @elapsed begin
            ensemble_rbon_fit!(ensemble, U_train, V_train, y_data, α, λ, use_mlp = false)
        end
        
        # Evaluate the model
        mean_error, std_error, mse, mae, predictions = ensemble_rbon_evaluate(ensemble, U_test, V_test, y_data)
        
        # Store results
        push!(results, (
            dataset = dataset_name,
            n_models = n_models,
            fit_time = fit_time,
            mean_error = mean_error,
            std_error = std_error,
            mse = mse,
            mae = mae
        ))
        
        println("    Fit time: $(@sprintf("%.2f", fit_time)) seconds")
        println("    Test Error: $(@sprintf("%.6f", mean_error)) ± $(@sprintf("%.6f", std_error))")
        println()
    end
    println()
end

# Print summary table
println("="^80)
println("Summary Table")
println("="^80)
println()

# Header
@printf("%-15s %-10s %-15s %-20s %-15s\n", "Dataset", "n_models", "Fit Time (s)", "Mean Error ± Std", "MAE")
println("-"^80)

# Print results
for r in results
    @printf("%-15s %-10d %-15.2f %-20s %-15.6f\n", 
        r.dataset, 
        r.n_models, 
        r.fit_time,
        "$(@sprintf("%.6f", r.mean_error)) ± $(@sprintf("%.6f", r.std_error))",
        r.mae
    )
end

println()
println("="^80)

# Save detailed results to a file
open("timing_results.txt", "w") do io
    println(io, "Ensemble RBON Training Time Results")
    println(io, "="^80)
    println(io)
    
    for r in results
        println(io, "Dataset: $(r.dataset)")
        println(io, "n_models: $(r.n_models)")
        println(io, "Fit Time: $(r.fit_time) seconds")
        println(io, "Mean Error: $(r.mean_error) ± $(r.std_error)")
        println(io, "MSE: $(r.mse)")
        println(io, "MAE: $(r.mae)")
        println(io, "-"^80)
    end
end

println("Results saved to timing_results.txt")

