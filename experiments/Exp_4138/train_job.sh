#!/bin/bash
#SBATCH --job-name=Exp_4138_train
#SBATCH --output=experiments/Exp_4138/output.log
#SBATCH --error=experiments/Exp_4138/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=33:20:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=5

echo "Starting training for experiment 4138"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3 4 -e 4138 -d "/home/epigou/scratch/metrics/Exp" 
    