#!/bin/bash
#SBATCH --job-name=Exp_206
#SBATCH --output=experiments/Exp_206/output.log
#SBATCH --error=experiments/Exp_206/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=05:39:59
#SBATCH --mem=64G

echo "Starting training for experiment 206"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g0 -e 206 -d "/home/hartman/scratch/metrics/Exp -eval True"
    