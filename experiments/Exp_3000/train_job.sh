#!/bin/bash
#SBATCH --job-name=Exp_3000_train
#SBATCH --output=experiments/Exp_3000/output.log
#SBATCH --error=experiments/Exp_3000/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --mem=180G

echo "Starting training for experiment 3000"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3 -e 3000 -d "/home/epigou/scratch/metrics/Exp"