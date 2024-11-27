#!/bin/bash
#SBATCH --job-name=Exp_360_train
#SBATCH --output=experiments/Exp_360/output.log
#SBATCH --error=experiments/Exp_360/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=05:20:00
#SBATCH --mem=16G


echo "Starting training for experiment 360"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 360 -d "/home/sgomezro/scratch/metrics/Exp" 
    