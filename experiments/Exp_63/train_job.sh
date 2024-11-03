#!/bin/bash
#SBATCH --job-name=Exp_63_train
#SBATCH --output=experiments/Exp_63/output.log
#SBATCH --error=experiments/Exp_63/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=0
#SBATCH --time=00:00:00
#SBATCH --mem=8G

echo "Starting training for experiment 63"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 63 -d "/home/hartman/scratch/metrics/Exp" 
    