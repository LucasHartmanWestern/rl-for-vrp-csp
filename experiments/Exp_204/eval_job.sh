#!/bin/bash
#SBATCH --job-name=Exp_204
#SBATCH --output=experiments/Exp_204/output.log
#SBATCH --error=experiments/Exp_204/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=0
#SBATCH --time=00:00:00
#SBATCH --mem=8G

echo "Starting training for experiment 204"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 204 -d "/home/hartman/scratch/metrics/Exp -eval True"
    