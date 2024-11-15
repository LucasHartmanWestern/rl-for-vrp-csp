#!/bin/bash
#SBATCH --job-name=Exp_138_train
#SBATCH --output=experiments/Exp_138/output.log
#SBATCH --error=experiments/Exp_138/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=41:40:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1

echo "Starting training for experiment 138"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 138 -d "/home/hartman/scratch/metrics/Exp" 
    