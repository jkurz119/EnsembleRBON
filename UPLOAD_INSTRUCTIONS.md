# Upload Instructions for GitHub Repository

## ✅ Files Ready to Upload

All code files are ready with correct paths! The `Ensemble_repo` directory contains:

### Code Files (Ready)
- ✅ `main.jl` - Main script with updated paths
- ✅ `EnsembleRBON.jl` - Core implementation with updated paths
- ✅ `src/RBON.jl` - RBON base class
- ✅ `src/RBON_ElasticNet.jl` - ElasticNet fitting
- ✅ `src/utils.jl` - Utility functions

### Configuration Files (Ready)
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules
- ✅ `Project.toml` - Julia dependencies

### Data Files (⚠️ YOU NEED TO COPY THESE)
- ⚠️ See `DATA_FILES_NEEDED.txt` for the list of required .mat files
- Copy all .mat files to the `data/` directory structure as shown

## Steps to Upload

1. **Copy Data Files:**
   ```powershell
   # Copy files from your original locations to Ensemble_repo/data/
   Copy-Item "EIT_4\*.mat" "Ensemble_repo\data\EIT_4\"
   Copy-Item "Ensemble\EIT_2_CoarseSamples.mat" "Ensemble_repo\data\Ensemble\"
   Copy-Item "Ensemble\CoarseGridPoints.mat" "Ensemble_repo\data\Ensemble\"
   ```

2. **Open Terminal/Command Prompt:**
   - On Windows: Open PowerShell or Command Prompt
   - On Mac/Linux: Open Terminal
   - Navigate to the Ensemble_repo directory:
     ```bash
     cd path\to\Ensemble_repo
     ```
     (Replace `path\to\Ensemble_repo` with the actual path to your Ensemble_repo folder)

3. **Initialize Git (if not already done):**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Ensemble RBON implementation"
   ```
   (Run these commands in the terminal while in the Ensemble_repo directory)

4. **Connect to GitHub and Push:**
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```
   (Replace `yourusername` and `your-repo-name` with your actual GitHub username and repository name)

## Directory Structure

```
Ensemble_repo/
├── main.jl
├── EnsembleRBON.jl
├── README.md
├── .gitignore
├── Project.toml
├── DATA_FILES_NEEDED.txt
├── UPLOAD_INSTRUCTIONS.md (this file)
├── data/
│   ├── EIT_4/
│   │   ├── EIT_FineSamples1.mat (copy here)
│   │   ├── EIT_FineSamples2.mat (copy here)
│   │   ├── EIT_FineSamples1or2.mat (copy here)
│   │   └── FineGridPoints.mat (copy here)
│   └── Ensemble/
│       ├── EIT_2_CoarseSamples.mat (copy here)
│       └── CoarseGridPoints.mat (copy here)
└── src/
    ├── RBON.jl ✅
    ├── RBON_ElasticNet.jl ✅
    └── utils.jl ✅
```

## Testing Before Upload

Test locally first:
```julia
cd Ensemble_repo
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -t 20 main.jl
```

If this works, you're ready to upload!

