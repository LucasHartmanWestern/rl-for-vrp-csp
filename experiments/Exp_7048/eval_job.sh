#!/bin/bash
#SBATCH --job-name=Exp_7048_eval
#SBATCH --output=experiments/Exp_7048/output.log
#SBATCH --error=experiments/Exp_7048/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=133:00:00
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting evaluation for experiment 7048"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 7048 -d "/home/hartman/scratch/metrics/Exp" -eval True
    