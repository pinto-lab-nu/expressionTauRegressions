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
conda activate allenSDKenv

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
        -h|--help) 
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -g, --gene_limit <gene_limit>"
            echo "  -l, --line_selection <line_selection>"
            echo "  -r, --restrict_merfish_imputed_values <restrict_merfish_imputed_values>"
            echo "  -t, --tau_pool_size_array_full <tau_pool_size_array_full>"
            echo "  -n, --n_splits <n_splits>"
            echo "  -a, --alpha_params <alpha_params>"
            echo "  -p, --plotting <plotting>"
            echo "  -u, --num_precision <num_precision>"
            echo "  -e, --alpha_precision <alpha_precision>"
            echo "  -v, --verbose <verbose>"
            echo "  -o, --predictor_order <predictor_order>"
            echo "  -s, --regressions_to_start <regressions_to_start>"
            echo "  -m, --max_iter <max_iter>"
            echo "  -c, --variable_management <variable_management>"
            echo "  -k, --plotting_conditions <plotting_conditions>"
            echo "  -x, --arg_parse_test <arg_parse_test>"
            echo "  -h, --help"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Default values if not provided
gene_limit=${gene_limit:--1}
line_selection=${line_selection:-"Cux2-Ai96"}
restrict_merfish_imputed_values=${restrict_merfish_imputed_values:-false}
tau_pool_size_array_full=${tau_pool_size_array_full:-4.1}
n_splits=${n_splits:-5}
alpha_params=${alpha_params:--5,0,30}
plotting=${plotting:-true}
num_precision=${num_precision:-3}
alpha_precision=${alpha_precision:-5}
verbose=${verbose:-true}
predictor_order=${predictor_order:-0}
regressions_to_start=${regressions_to_start:-0,1}
max_iter=${max_iter:-200}
variable_management=${variable_management:-true}
plotting_conditions=${plotting_conditions:-0,1} # converted to bool in python script
arg_parse_test=${arg_parse_test:-false}

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
    --job_task_id=$SLURM_ARRAY_TASK_ID
