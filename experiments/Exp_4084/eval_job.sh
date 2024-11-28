#!/bin/bash
#SBATCH --job-name=Exp_4084_eval
#SBATCH --output=experiments/Exp_4084/output.log
#SBATCH --error=experiments/Exp_4084/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --mem=16G


echo "Starting evaluation for experiment 4084"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 4084 -d "/home/sgomezro/scratch/metrics/Exp" -eval True
    