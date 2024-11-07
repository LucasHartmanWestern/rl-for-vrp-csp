#!/bin/bash
#SBATCH --job-name=Exp_15_train
#SBATCH --output=experiments/Exp_15/output.log
#SBATCH --error=experiments/Exp_15/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --time=50:00:00
#SBATCH --mem=24G

echo "Starting training for experiment 15"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 15 -d "/home/hartman/scratch/metrics/Exp" 
    