#!/bin/bash
#SBATCH --job-name=Exp_4063_train
#SBATCH --output=experiments/Exp_4063/output.log
#SBATCH --error=experiments/Exp_4063/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=80:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

echo "Starting training for experiment 4063"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 4063 -d "/home/hartman/scratch/metrics/Exp" 
    