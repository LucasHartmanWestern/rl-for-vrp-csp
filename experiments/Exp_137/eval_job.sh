#!/bin/bash
#SBATCH --job-name=Exp_137_eval
#SBATCH --output=experiments/Exp_137/output.log
#SBATCH --error=experiments/Exp_137/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=24G

echo "Starting evaluation for experiment 137"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 137 -d "/home/hartman/scratch/metrics/Exp" -eval True
    