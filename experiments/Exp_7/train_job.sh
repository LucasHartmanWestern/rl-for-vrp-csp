#!/bin/bash
#SBATCH --job-name=Exp_7
#SBATCH --output=experiments/Exp_7/output.log
#SBATCH --error=experiments/Exp_7/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=3
#SBATCH --time=23:59:59
#SBATCH --mem=64G

echo "Starting training for experiment 7"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=6

python app_v2.py -g 0 1 2 -e 7 -d "/home/hartman/scratch/metrics/Exp"
