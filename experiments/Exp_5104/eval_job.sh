#!/bin/bash
#SBATCH --job-name=Exp_5104_eval
#SBATCH --output=experiments/Exp_5104/output.log
#SBATCH --error=experiments/Exp_5104/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:22:30
#SBATCH --mem=6G


echo "Starting evaluation for experiment 5104"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 5104 -d "/home/sgomezro/scratch/metrics/Exp" -eval True
    