#!/bin/bash
#SBATCH --job-name=Exp_80_eval
#SBATCH --output=experiments/Exp_80/output.log
#SBATCH --error=experiments/Exp_80/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=3
#SBATCH --time=15:00:00
#SBATCH --mem=64G

echo "Starting evaluation for experiment 80"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 -e 80 -d "/home/hartman/scratch/metrics/Exp" -eval True
    