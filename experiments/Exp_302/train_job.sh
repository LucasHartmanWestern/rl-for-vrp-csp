#!/bin/bash
#SBATCH --job-name=Exp_302_train
#SBATCH --output=experiments/Exp_302/output.log
#SBATCH --error=experiments/Exp_302/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --mem=160G

echo "Starting training for experiment 302"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3-e 302 -d "/home/epigou/scratch/metrics/Exp" 
    