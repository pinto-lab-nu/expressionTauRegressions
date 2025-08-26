# Expression-Tau Regression Analysis

## Overview

This repository provides tools for analyzing the relationship between gene expression patterns and neural functional timescales (tau) in cortical brain regions. The project integrates two complementary datasets:

- **MERFISH spatial transcriptomics data** from the Allen Institute Brain Cell Atlas
- **Pilot dataset** containing neural tau measurements from widefield calcium imaging

The core functionality performs regularized regression analysis (primarily Lasso regression) to predict neural timescales from gene expression profiles across different cortical layers and brain regions. The analysis is spatially-aware, utilizing the Allen Brain Atlas Common Coordinate Framework (CCF) for precise anatomical registration.

**Key Applications:**
- Identify gene expression patterns that predict neural dynamics
- Understand layer-specific relationships between transcription and function
- Spatial mapping of gene-function relationships across cortical areas
- Cross-validation of gene expression predictors of neural timescales

## Quick Start

For users who want to get started immediately:

```bash
# 1. Set up environment
conda env create -f requirements.yml
conda activate allenSDKenv

# 2. Test with small dataset
python expressionRegression.py --gene_limit 10 --plotting false --preprocessing_only true

# 3. Run basic analysis
python expressionRegression.py --gene_limit 100 --line_selection "Cux2-Ai96" --n_splits 3

# 4. For full analysis (requires HPC)
sbatch regressionJob.sh --gene_limit -1 --plotting true
```

## Installation & Usage

### Prerequisites

This project requires Python 3.11 and several scientific computing packages. The analysis is designed to run on high-performance computing clusters with SLURM job scheduling.

### Environment Setup

1. **Create conda environment from requirements:**
   ```bash
   conda env create -f requirements.yml
   conda activate allenSDKenv
   ```

2. **Install additional dependencies:**
   The environment includes custom packages that may need manual installation:
   - `ccfRegistration` - Custom package for brain atlas registration (contact repository authors)
   - Allen SDK and ABC Atlas Access packages (automatically installed via pip from git)

3. **Data Requirements:**
   - The analysis requires access to Allen Institute datasets which are automatically downloaded
   - Functional tau data must be provided separately or generated using the database integration
   - Sufficient storage space (several GB) for cached datasets

### Basic Usage

#### Interactive Analysis
```bash
python expressionRegression.py --line_selection "Cux2-Ai96" --gene_limit 100 --plotting true
```

#### High-Performance Computing
For large-scale analysis on SLURM clusters:
```bash
# Submit regression job with custom parameters
sbatch regressionJob.sh --gene_limit -1 --n_splits 5 --plotting true

# Submit with specific alpha parameters for Lasso regression
sbatch regressionJob.sh --alpha_params "-5,2,30" --max_iter 500
```

#### Quick Test Run
```bash
# Test with limited genes and quick parameters
python expressionRegression.py --gene_limit 10 --arg_parse_test false --preprocessing_only true
```

## Key Features

- **Multi-modal Integration:** Combines spatial transcriptomics (MERFISH) with functional imaging data
- **Spatial Awareness:** Uses Allen Brain Atlas CCF for precise anatomical registration
- **Robust Statistics:** Cross-validated Lasso regression with comprehensive model evaluation
- **Layer-specific Analysis:** Separate analysis for different cortical layers
- **HPC Ready:** Optimized for high-performance computing with SLURM integration
- **Comprehensive Visualization:** Publication-quality plots and spatial maps
- **Memory Efficient:** Handles large datasets with intelligent memory management

## Table of Contents

### Core Modules

#### `expressionRegression.py`
**Main analysis script that orchestrates the entire regression pipeline.**

- **Purpose:** Coordinates data loading, preprocessing, regression analysis, and visualization
- **Key Features:**
  - Command-line interface with extensive parameter configuration
  - Support for multiple regression types (gene→tau, gene→CCF, combined analyses)
  - Automated spatial registration using Allen Brain Atlas
  - Memory management for large datasets
- **Important Parameters:**
  - `--line_selection`: Choose functional dataset ("Cux2-Ai96" or "Rpb4-Ai96")
  - `--gene_limit`: Limit genes for testing (-1 for all genes)
  - `--alpha_params`: Lasso regularization parameters as "min,max,steps"
  - `--n_splits`: Cross-validation folds (default: 5)
- **Design Notes:** Modular architecture allows easy extension for new datasets or analysis types

#### `packages/dataloading.py`
**Handles data acquisition and preprocessing from multiple sources.**

- **Purpose:** Load and preprocess MERFISH and pilot datasets with proper spatial alignment
- **Key Functions:**
  - `merfishLoader()`: Downloads and processes Allen Institute MERFISH data
  - `pilotLoader()`: Loads pilot dataset with tau measurements
  - `pathSetter()`: Cross-platform path management for Linux/Windows HPC environments
- **Implementation Details:**
  - Uses Allen Brain Institute ABC Atlas Cache for automatic data download
  - Handles both imputed and non-imputed gene expression values
  - Applies standardization and filtering based on gene expression thresholds
- **Reusable Features:** Modular loading functions can be adapted for other spatial transcriptomics datasets

#### `packages/regressionUtils.py`
**Core machine learning functionality for regularized regression analysis.**

- **Purpose:** Perform cross-validated Lasso regression with comprehensive model evaluation
- **Key Functions:**
  - `layerRegressions()`: Main regression function with layer-wise analysis
  - `predictor_response_info()`: Data shape validation and statistics
  - `PDFmerger()`: Combine multiple result plots into single documents
- **Implementation Details:**
  - Supports both single-task and multi-task Lasso regression
  - Comprehensive cross-validation with configurable alpha grid search
  - Loss landscape tracking for convergence monitoring
  - Memory-efficient handling of large prediction matrices
- **Advanced Features:**
  - Dual gap monitoring for optimization convergence
  - Beta coefficient tracking across regularization strengths
  - Support for weighted regression (preparation for uncertainty quantification)

#### `packages/plotting_util.py`
**Comprehensive visualization suite for regression results and spatial data.**

- **Purpose:** Generate publication-quality plots for regression analysis and spatial visualization
- **Key Functions:**
  - `plot_regressions()`: Main plotting function with extensive customization options
  - Spatial reconstruction plots with CCF overlay
  - Cross-layer correlation analysis
  - Beta coefficient visualization across brain regions
- **Plot Types:**
  - R² performance across regularization parameters
  - Spatial maps of prediction accuracy
  - Gene coefficient significance plots
  - Cross-layer beta correlation matrices
- **Design Features:**
  - Publication-ready formatting with customizable fonts and colors
  - Automated plot saving with organized directory structure
  - Support for both individual plots and summary documents

#### `packages/functional_dataset.py`
**Framework for functional timescale data integration (development/future use).**

- **Purpose:** Handle functional imaging data and tau calculations
- **Current Status:** Primarily contains commented-out code for future database integration
- **Potential Features:**
  - Direct database connectivity for functional imaging data
  - Automated tau calculation from calcium imaging timeseries
  - Cross-modal data registration and validation

### Utility Scripts

#### `geneCategories.py`
**Predefined gene classification system for targeted analysis.**

- **Purpose:** Organize genes into functional categories for hypothesis-driven analysis
- **Categories Include:**
  - Ion channels and ion homeostasis
  - Neurotransmitter receptors and transporters
  - Metabolic genes
  - Transcription factors
- **Usage:** Import specific gene lists for focused regression analysis or validation studies

#### `regressionJob.sh`
**Production SLURM job submission script with comprehensive parameter handling.**

- **Purpose:** Submit regression analyses to HPC clusters with proper resource allocation
- **Features:**
  - Extensive command-line argument parsing
  - Default parameter values for standard analyses
  - Memory and time allocation suitable for large datasets
  - Array job support for parallel processing
- **Resource Requirements:**
  - Default: 180GB RAM, 11-hour time limit
  - Configurable for different dataset sizes and analysis complexity

#### `merger.py` and `cleaningJob.sh`
**Post-processing utilities for result organization and cleanup.**

- **Purpose:** Organize and clean up large numbers of result files
- **Features:**
  - Automated file merging and organization
  - Cleanup of temporary files and intermediate results
  - Integration with HPC job scheduling for automated post-processing

## Examples

### Basic Regression Analysis
```python
# Example: Run regression analysis on a subset of genes
import os
from packages.dataloading import pathSetter, merfishLoader, pilotLoader
from packages.regressionUtils import layerRegressions
from packages.plotting_util import plot_regressions

# Set up paths and load data
my_os, save_path, download_base = pathSetter()
gene_data, gene_names, clusters, ccf_coords = pilotLoader(save_path)

# Configure regression parameters
alpha_params = [-5, -2, 30]  # 10^-5 to 10^-2, 30 steps
n_splits = 5
layer_names = ['L2_3 IT', 'L4_5 IT', 'L5 IT', 'L6 IT', 'L5 ET']

# Run regression analysis
best_betas, betas_over_alpha, best_alpha, alphas, predictions, best_r2, \
loss_test, loss_train, dual_gap = layerRegressions(
    response_dim=1,
    n_splits=n_splits,
    highMeanPredictorIDXs=[range(gene_data.shape[1])],  # Use all genes
    x_data=[gene_data],
    y_data=[tau_data],
    layerNames=layer_names,
    regressionConditions=[True, False, False, True],
    cell_region={},
    alphaParams=alpha_params,
    max_iter=200,
    verbose=True
)
```

### Spatial Visualization
```python
# Example: Create spatial plots of regression results
plot_regressions(
    generate_plots=True,
    generate_summary=True,
    lineSelection="Cux2-Ai96",
    structList=['MOp', 'MOs', 'VISp', 'VISa'],
    areaColors=['#1B50FF', '#2F4077', '#4dd2ff', '#0066ff'],
    plottingConditions=[True],
    params={'alpha_precision': 5, 'num_precision': 3},
    paths={'plotting_dir': '/path/to/plots'},
    titles={'dataset': 'MERFISH-Imputed'},
    model_vals={'best_r2': best_r2, 'best_alpha': best_alpha},
    plotting_data={'predictions': predictions},
    paper_plotting=True
)
```

### Command-Line Usage
```bash
# Comprehensive analysis with spatial plotting
python expressionRegression.py \
    --line_selection "Cux2-Ai96" \
    --gene_limit -1 \
    --restrict_merfish_imputed_values false \
    --tau_pool_size_array_full "4.0" \
    --n_splits 5 \
    --alpha_params "-5,-2,30" \
    --plotting true \
    --regressions_to_start "0,1,2,3" \
    --max_iter 200 \
    --verbose true

# Quick test with limited genes
python expressionRegression.py \
    --gene_limit 50 \
    --plotting false \
    --preprocessing_only true \
    --verbose true
```

### HPC Cluster Submission
```bash
# Submit to SLURM cluster with custom parameters
sbatch --array=0-4%1 regressionJob.sh \
    --gene_limit -1 \
    --line_selection "Cux2-Ai96" \
    --alpha_params "-5,2,40" \
    --n_splits 10 \
    --plotting true

# Monitor job progress
squeue -u $USER
tail -f /mnt/fsmresfiles/Tau_Processing/H3/SlurmLogs/regression_*.txt
```

## Troubleshooting

### Common Issues

1. **Memory Errors:**
   - Reduce `gene_limit` for testing: `--gene_limit 100`
   - Use `--variable_management true` to free memory after use
   - Request more memory in SLURM: modify `#SBATCH --mem=` in regressionJob.sh

2. **Data Download Issues:**
   - Ensure internet connectivity for Allen Institute data download
   - Check available disk space (several GB required)
   - Verify Allen SDK installation: `pip install git+https://github.com/alleninstitute/allensdk`

3. **Missing ccfRegistration Package:**
   - Contact repository authors for access to custom registration package
   - This package handles brain atlas coordinate transformations

4. **SLURM Job Failures:**
   - Check log files in `/mnt/fsmresfiles/Tau_Processing/H3/SlurmLogs/`
   - Verify conda environment is activated: `conda activate allenSDKenv`
   - Ensure proper file permissions on shared storage

### Performance Tips

- Use `--preprocessing_only true` for initial data setup
- Start with small gene limits (`--gene_limit 50`) for testing
- Monitor memory usage with `--verbose true`
- Use array jobs for parallel processing: `--array=0-4%1`

---

**Repository Structure:**
```
expressionTauRegressions/
├── expressionRegression.py     # Main analysis script
├── packages/
│   ├── dataloading.py         # Data loading and preprocessing
│   ├── regressionUtils.py     # Machine learning utilities
│   ├── plotting_util.py       # Visualization functions
│   └── functional_dataset.py  # Functional data handling
├── geneCategories.py          # Gene classification definitions
├── regressionJob.sh          # HPC job submission script
├── cleaningJob.sh            # Post-processing cleanup
├── merger.py                 # Result file organization
└── requirements.yml          # Conda environment specification
```

**Citation:** If you use this code in your research, please cite the associated publication and acknowledge the Allen Institute for Brain Science for the MERFISH dataset.