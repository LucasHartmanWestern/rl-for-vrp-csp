#!/bin/bash
#SBATCH --job-name=Exp_4122_eval
#SBATCH --output=experiments/Exp_4122/output.log
#SBATCH --error=experiments/Exp_4122/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=33:20:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=5

echo "Starting evaluation for experiment 4122"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3 4 -e 4122 -d "/home/epigou/scratch/metrics/Exp" -eval True
    