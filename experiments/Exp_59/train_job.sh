#!/bin/bash
#SBATCH --job-name=Exp_59
#SBATCH --output=experiments/Exp_59/output.log
#SBATCH --error=experiments/Exp_59/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=5
#SBATCH --time=40:00:00
#SBATCH --mem=64G

echo "Starting training for experiment 59"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=10

python app_v2.py -g 0 1 2 3 4 -e 59 -d "/home/hartman/scratch/metrics/Exp"
