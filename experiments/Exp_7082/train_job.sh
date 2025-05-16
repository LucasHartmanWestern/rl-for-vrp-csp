#!/bin/bash
#SBATCH --job-name=Exp_7082_train
#SBATCH --output=experiments/Exp_7082/output.log
#SBATCH --error=experiments/Exp_7082/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:40:00
#SBATCH --mem=6G


echo "Starting training for experiment 7082"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 7082 -d "/home/sgomezro/scratch/metrics/Exp" 
    