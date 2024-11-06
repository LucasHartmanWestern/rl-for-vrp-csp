#!/bin/bash
#SBATCH --job-name=Exp_90_train
#SBATCH --output=experiments/Exp_90/output.log
#SBATCH --error=experiments/Exp_90/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=24G

echo "Starting training for experiment 90"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 90 -d "/home/hartman/scratch/metrics/Exp" 
    