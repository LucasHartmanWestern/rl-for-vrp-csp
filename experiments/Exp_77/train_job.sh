#!/bin/bash
#SBATCH --job-name=Exp_77_train
#SBATCH --output=experiments/Exp_77/output.log
#SBATCH --error=experiments/Exp_77/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --time=15:00:00
#SBATCH --mem=24G

echo "Starting training for experiment 77"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 77 -d "/home/hartman/scratch/metrics/Exp" 
    