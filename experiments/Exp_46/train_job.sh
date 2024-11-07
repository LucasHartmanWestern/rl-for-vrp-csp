#!/bin/bash
#SBATCH --job-name=Exp_46_train
#SBATCH --output=experiments/Exp_46/output.log
#SBATCH --error=experiments/Exp_46/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=26:39:59
#SBATCH --mem=24G

echo "Starting training for experiment 46"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 46 -d "/home/hartman/scratch/metrics/Exp" 
    