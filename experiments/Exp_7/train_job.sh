#!/bin/bash
#SBATCH --job-name=Exp_7_train
#SBATCH --output=experiments/Exp_7/output.log
#SBATCH --error=experiments/Exp_7/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --mem=24G

echo "Starting training for experiment 7"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 7 -d "/home/hartman/scratch/metrics/Exp" 
    