#!/bin/bash
#SBATCH --job-name=Exp_78_eval
#SBATCH --output=experiments/Exp_78/output.log
#SBATCH --error=experiments/Exp_78/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=3
#SBATCH --time=15:00:00
#SBATCH --mem=64G

echo "Starting evaluation for experiment 78"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g0 1 2 -e 2 -d "/home/hartman/scratch/metrics/Exp" -eval True
    