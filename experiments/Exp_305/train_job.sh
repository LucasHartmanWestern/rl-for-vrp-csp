#!/bin/bash
#SBATCH --job-name=Exp_305_train
#SBATCH --output=experiments/Exp_305/output.log
#SBATCH --error=experiments/Exp_305/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --mem=160G

echo "Starting training for experiment 305"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3-e 305 -d "/home/epigou/scratch/metrics/Exp" 
    