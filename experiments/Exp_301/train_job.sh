#!/bin/bash
#SBATCH --job-name=Exp_301_train
#SBATCH --output=experiments/Exp_301/output.log
#SBATCH --error=experiments/Exp_301/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=31:59:59
#SBATCH --mem=64G

echo "Starting training for experiment 301"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g0 -e 301 -d "/home/epigou/scratch/metrics/Exp" 
    