#!/bin/bash
#SBATCH --job-name=Exp_4135_train
#SBATCH --output=experiments/Exp_4135/output.log
#SBATCH --error=experiments/Exp_4135/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=20:00:00
#SBATCH --mem=25G
#SBATCH --gpus-per-node=4

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 4135"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3 -e 4135 -d "/home/epigou/scratch/metrics/Exp"