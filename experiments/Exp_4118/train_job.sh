#!/bin/bash
#SBATCH --job-name=Exp_4118_train
#SBATCH --output=experiments/Exp_4118/output.log
#SBATCH --error=experiments/Exp_4118/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=13:30:00
#SBATCH --mem=15G
#SBATCH --gpus-per-node=4

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 4118"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3 -e 4118 -d "/home/epigou/scratch/metrics/Exp"
