#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=tauRegression
#SBATCH --mem=75GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/fsmresfiles/Tau_Processing/H3/SlurmLogs/regression_%j.txt
#SBATCH --array=0-4%1

export PYTHONUNBUFFERED=1 #for real-time log files

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchEnv #allenSDKenv

# Parse command line arguments for all variables
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gene_limit) gene_limit="$2"; shift 2 ;;
        -l|--line_selection) line_selection="$2"; shift 2 ;;
        -r|--restrict_merfish_imputed_values) restrict_merfish_imputed_values="$2"; shift 2 ;;
        -t|--tau_pool_size_array_full) tau_pool_size_array_full="$2"; shift 2 ;;
        -n|--n_splits) n_splits="$2"; shift 2 ;;
        -a|--alpha_params) alpha_params="$2"; shift 2 ;;
        -p|--plotting) plotting="$2"; shift 2 ;;
        -u|--num_precision) num_precision="$2"; shift 2 ;;
        -e|--alpha_precision) alpha_precision="$2"; shift 2 ;;
        -v|--verbose) verbose="$2"; shift 2 ;;
        -o|--predictor_order) predictor_order="$2"; shift 2 ;;
        -s|--regressions_to_start) regressions_to_start="$2"; shift 2 ;;
        -m|--max_iter) max_iter="$2"; shift 2 ;;
        -c|--variable_management) variable_management="$2"; shift 2 ;;
        -k|--plotting_conditions) plotting_conditions="$2"; shift 2 ;;
        -x|--arg_parse_test) arg_parse_test="$2"; shift 2 ;;
        -b|--bootstrapping_scale) bootstrapping_scale="$2"; shift 2 ;;
        -i|--min_pool_size) min_pool_size="$2"; shift 2 ;;
        -h|--help) 
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -g, --gene_limit                        # For testing purposes to load a subset of merfish-imputed data, set to -1 to include all genes"
            echo "  -l, --line_selection                    # Select the functional dataset for tau regressions"
            echo "  -r, --restrict_merfish_imputed_values   # Condition to restrict merfish-imputed dataset to non-imputed genes"
            echo "  -t, --tau_pool_size_array_full          # In 25um resolution CCF voxels, converted to mm later"
            echo "  -n, --n_splits                          # Number of splits for cross-validations in regressions"
            echo "  -a, --alpha_params                      # Alpha values for Lasso regressions"
            echo "  -p, --plotting                          # Enable or disable plotting"
            echo "  -u, --num_precision                     # Just for display (in plotting and regression text files)"
            echo "  -e, --alpha_precision                   # Just for display (in plotting and regression text files)"
            echo "  -v, --verbose                           # For print statements"
            echo "  -o, --predictor_order                   # Select predictors for regressions, and order"
            echo "  -s, --regressions_to_start              # Select response variables for regressions, and order"
            echo "  -m, --max_iter                          # For layer regressions"
            echo "  -c, --variable_management               # Removes large variables from memory after use"
            echo "  -k, --plotting_conditions               # For plotting spatial reconstructions"
            echo "  -x, --arg_parse_test                    # For testing the bash argument parser"
            echo "  -b, --bootstrapping_scale               # Scale for bootstrapping, default is 1.0"
            echo "  -i, --min_pool_size                     # Minimum pool size for tau regressions, default is 3"
            echo "  -h, --help                              # Display this help message"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
        esac
    done

# Default values if not provided
gene_limit=${gene_limit:--1}
line_selection=${line_selection:-"Cux2-Ai96"}
restrict_merfish_imputed_values=${restrict_merfish_imputed_values:-false}
tau_pool_size_array_full=${tau_pool_size_array_full:-4.0}
n_splits=${n_splits:-5}
alpha_params=${alpha_params:--5,0,30}
plotting=${plotting:-true}
num_precision=${num_precision:-3}
alpha_precision=${alpha_precision:-5}
verbose=${verbose:-true}
predictor_order=${predictor_order:-0}
regressions_to_start=${regressions_to_start:-0,1,2,3}
max_iter=${max_iter:-200}
variable_management=${variable_management:-true}
plotting_conditions=${plotting_conditions:-0,1} # converted to bool in python script
arg_parse_test=${arg_parse_test:-false}
bootstrapping_scale=${bootstrapping_scale:-1.0}
min_pool_size=${min_pool_size:-3}

python expressionRegression.py \
    --gene_limit=$gene_limit \
    --line_selection=$line_selection \
    --restrict_merfish_imputed_values=$restrict_merfish_imputed_values \
    --tau_pool_size_array_full=$tau_pool_size_array_full \
    --n_splits=$n_splits \
    --alpha_params=$alpha_params \
    --plotting=$plotting \
    --num_precision=$num_precision \
    --alpha_precision=$alpha_precision \
    --verbose=$verbose \
    --predictor_order=$predictor_order \
    --regressions_to_start=$regressions_to_start \
    --max_iter=$max_iter \
    --variable_management=$variable_management \
    --plotting_conditions=$plotting_conditions \
    --arg_parse_test=$arg_parse_test \
    --job_task_id=$SLURM_ARRAY_TASK_ID \
    --bootstrapping_scale=$bootstrapping_scale \
    --min_pool_size=$min_pool_size \

# Example usage:
# sbatch --array=0-0%1 regressionJob.sh --arg_parse_test true --alphaParams -5,5,30

# bash regressionJob.sh -h