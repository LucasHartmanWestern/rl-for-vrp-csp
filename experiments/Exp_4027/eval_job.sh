#!/bin/bash
#SBATCH --job-name=Exp_4027_eval
#SBATCH --output=experiments/Exp_4027/output.log
#SBATCH --error=experiments/Exp_4027/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=90:00:00
#SBATCH --mem=24G


echo "Starting evaluation for experiment 4027"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 4027 -d "/home/hartman/scratch/metrics/Exp" -eval True
    