#!/bin/bash
#SBATCH --job-name=Exp_5_train
#SBATCH --output=experiments/Exp_5/output.log
#SBATCH --error=experiments/Exp_5/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --time=15:00:00
#SBATCH --mem=24G

echo "Starting training for experiment 5"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 5 -d "/home/hartman/scratch/metrics/Exp" 
    