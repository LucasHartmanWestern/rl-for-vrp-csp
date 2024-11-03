#!/bin/bash
#SBATCH --job-name=Exp_59
#SBATCH --output=experiments/Exp_59/output.log
#SBATCH --error=experiments/Exp_59/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=53:19:59
#SBATCH --mem=64G

echo "Starting training for experiment 59"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g0 1 2 3 -e 59 -d "/home/hartman/scratch/metrics/Exp"
    