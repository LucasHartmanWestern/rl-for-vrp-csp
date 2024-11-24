#!/bin/bash
#SBATCH --job-name=Exp_4098_train
#SBATCH --output=experiments/Exp_4098/output.log
#SBATCH --error=experiments/Exp_4098/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=13:00:00
#SBATCH --mem=16G


echo "Starting training for experiment 4098"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 4098 -d "/home/hartman/scratch/metrics/Exp" 
    