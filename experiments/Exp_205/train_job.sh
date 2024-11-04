#!/bin/bash
#SBATCH --job-name=Exp_205_train
#SBATCH --output=experiments/Exp_205/output.log
#SBATCH --error=experiments/Exp_205/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=0
#SBATCH --time=00:00:00
#SBATCH --mem=8G

echo "Starting training for experiment 205"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py   -e 205 -d "/home/hartman/scratch/metrics/Exp" 
    