#!/bin/bash
#SBATCH --job-name=Exp_57_eval
#SBATCH --output=experiments/Exp_57/output.log
#SBATCH --error=experiments/Exp_57/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=0
#SBATCH --time=00:00:00
#SBATCH --mem=8G

echo "Starting training for experiment 57"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 57 -d "/home/hartman/scratch/metrics/Exp" -eval True
    