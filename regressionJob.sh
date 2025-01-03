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

#['Rpb4-Ai96','Cux2-Ai96','C57BL6/J'] #just here for the convenience of line selection indexing (handled in expressionRegression.py)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate allenSDKenv

# Parse command line arguments for all variables
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--geneLimit) geneLimit="$2"; shift ;;
        -l|--lineSelection) lineSelection="$2"; shift ;;
        -r|--restrict_merfish_imputed_values) restrict_merfish_imputed_values="$2"; shift ;;
        -t|--tauPoolSizeArrayFull) tauPoolSizeArrayFull="$2"; shift ;;
        -n|--n_splits) n_splits="$2"; shift ;;
        -a|--alphaParams) alphaParams="$2"; shift ;;
        -d|--loadData) loadData="$2"; shift ;;
        -p|--plotting) plotting="$2"; shift ;;
        -u|--numPrecision) numPrecision="$2"; shift ;;
        -e|--alphaPrecision) alphaPrecision="$2"; shift ;;
        -v|--verbose) verbose="$2"; shift ;;
        -o|--predictorOrder) predictorOrder="$2"; shift ;;
        -s|--regressionsToStart) regressionsToStart="$2"; shift ;;
        -m|--max_iter) max_iter="$2"; shift ;;
        -c|--variableManagement) variableManagement="$2"; shift ;;
        -k|--plottingConditions) plottingConditions="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Default values if not provided
geneLimit=${geneLimit:--1}
lineSelection=${lineSelection:-"Cux2-Ai96"}
restrict_merfish_imputed_values=${restrict_merfish_imputed_values:-false}
tauPoolSizeArrayFull=${tauPoolSizeArrayFull:-4.1}
n_splits=${n_splits:-5}
alphaParams=${alphaParams:--5,0,30}
loadData=${loadData:-true}
plotting=${plotting:-true}
numPrecision=${numPrecision:-3}
alphaPrecision=${alphaPrecision:-5}
verbose=${verbose:-true}
predictorOrder=${predictorOrder:-0}
regressionsToStart=${regressionsToStart:-0,1}
max_iter=${max_iter:-200}
variableManagement=${variableManagement:-true}
plottingConditions=${plottingConditions:-false,true}

python expressionRegression.py \
    --geneLimit=$geneLimit \
    --lineSelection=$lineSelection \
    --restrict_merfish_imputed_values=$restrict_merfish_imputed_values \
    --tauPoolSizeArrayFull=$tauPoolSizeArrayFull \
    --n_splits=$n_splits \
    --alphaParams=$alphaParams \
    --loadData=$loadData \
    --plotting=$plotting \
    --numPrecision=$numPrecision \
    --alphaPrecision=$alphaPrecision \
    --verbose=$verbose \
    --predictorOrder=$predictorOrder \
    --regressionsToStart=$regressionsToStart \
    --max_iter=$max_iter \
    --variableManagement=$variableManagement \
    --plottingConditions=$plottingConditions \
    --job_task_id=$SLURM_ARRAY_TASK_ID
