#!/bin/bash
#SBATCH --job-name=Exp_303_train
#SBATCH --output=experiments/Exp_303/output.log
#SBATCH --error=experiments/Exp_303/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --mem=160G

echo "Starting training for experiment 303"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3 -e 303 -d "/home/epigou/scratch/metrics/Exp" 
    