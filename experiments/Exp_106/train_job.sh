#!/bin/bash
#SBATCH --job-name=Exp_106_train
#SBATCH --output=experiments/Exp_106/output.log
#SBATCH --error=experiments/Exp_106/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:20:00
#SBATCH --mem=24G


echo "Starting training for experiment 106"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py   -e 106 -d "/home/hartman/scratch/metrics/Exp" 
    