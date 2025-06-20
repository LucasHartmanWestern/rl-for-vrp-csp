#!/bin/bash
#SBATCH --job-name=Exp_350_train
#SBATCH --output=experiments/Exp_0350/output.log
#SBATCH --error=experiments/Exp_0350/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=30:00:00
#SBATCH --mem=32G


echo "Starting training for experiment 0350"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 0350 -d "/home/sgomezro/scratch/metrics/Exp" 
    