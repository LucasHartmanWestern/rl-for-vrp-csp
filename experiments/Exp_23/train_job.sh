#!/bin/bash
#SBATCH --job-name=Exp_23
#SBATCH --output=experiments/Exp_23/output.log
#SBATCH --error=experiments/Exp_23/error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=3
#SBATCH --time=55:00:00
#SBATCH --mem=128G

echo "Starting training for experiment 23"
nvidia-smi

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=6

python app_v2.py -g 0 1 2 -e 23 -d "/home/hartman/scratch/metrics/Exp"
