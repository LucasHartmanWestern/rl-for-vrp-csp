#!/bin/bash
#SBATCH --job-name=Exp_6066_eval
#SBATCH --output=experiments/Exp_6066/output.log
#SBATCH --error=experiments/Exp_6066/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=133:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1

echo "Starting evaluation for experiment 6066"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 6066 -d "/home/hartman/scratch/metrics/Exp" -eval True
    