#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=tauRegression
#SBATCH --mem=15GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/fsmresfiles/Tau_Processing/H3/SlurmLogs/regression_%j.txt
#SBATCH --array=0-3%4

export PYTHONUNBUFFERED=1 #for real-time log files

#['Rpb4-Ai96','Cux2-Ai96','C57BL6/J'] #just here for the convenience of line selection indexing (handled in expressionRegression.py)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate allenSDKenv

python expressionRegression.py $SLURM_ARRAY_TASK_ID 1 #last variable indicates line selection for analysis (see above)
