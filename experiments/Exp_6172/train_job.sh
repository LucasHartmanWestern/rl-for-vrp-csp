#!/bin/bash
#SBATCH --job-name=Exp_6172_train
#SBATCH --output=experiments/Exp_6172/output.log
#SBATCH --error=experiments/Exp_6172/error.log
#SBATCH -A  rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=22G

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 6172"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=8



python app_v2.py -e 6172 -d "/home/epigou/scratch/metrics/Exp"
