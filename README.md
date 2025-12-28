# Ensemble RBON

This repository contains an implementation of an ensemble of Radial Basis Operator Networks (RBON) for solving inverse problems, specifically applied to Electrical Impedance Tomography (EIT).

## Repository Structure

```
Ensemble/
├── main.jl                    # Main script to run timing comparisons
├── EnsembleRBON.jl           # Core ensemble RBON implementation
├── README.md                  # This file
├── data/                      # Data files directory
│   ├── EIT_4/
│   │   ├── EIT_FineSamples1.mat
│   │   ├── EIT_FineSamples2.mat
│   │   ├── EIT_FineSamples1or2.mat
│   │   └── FineGridPoints.mat
│   └── Ensemble/
│       ├── EIT_2_CoarseSamples.mat
│       └── CoarseGridPoints.mat
└── src/                       # Source code dependencies
    ├── RBON.jl
    ├── RBON_ElasticNet.jl
    └── utils.jl
```

## Dependencies

This project requires the following Julia packages (see `Project.toml`):
- MAT
- StatsBase
- Clustering
- Distances
- GLMNet
- Optim
- LinearAlgebra
- Statistics
- Base.Threads

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Ensemble
   ```

2. **Install Julia dependencies:**
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. **Ensure data files are in place:**
   - Place all `.mat` data files in the `data/` directory as shown in the structure above
   - Update paths in `main.jl` if your data is in a different location

## Usage

Run the main script to perform timing comparisons across different ensemble configurations:

```julia
julia --project=. -t 20 main.jl
```

The `-t 20` flag sets the number of threads (adjust based on your system).

## Output

The script will:
- Print timing results for different numbers of ensemble models (1, 5, 10, 15)
- Test on three different datasets (Anomaly1, Anomaly1or2, Anomaly2)
- Generate a summary table with fit times, mean errors, and MAE
- Save detailed results to `timing_results.txt`

## Results

The script tests ensemble RBON models with:
- **n_models**: 1, 5, 10, 15
- **Datasets**: Anomaly1, Anomaly1or2, Anomaly2
- **Metrics**: Fit time, relative error, MSE, MAE

## Notes

- The ensemble uses cosine similarity for clustering training data
- Each RBON in the ensemble uses 21x21 centers (h_num=21, l_num=21)
- Out-of-domain samples are detected and warnings are issued
- Results are saved to `timing_results.txt` for further analysis

