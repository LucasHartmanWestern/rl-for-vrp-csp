#!/bin/bash
#SBATCH --job-name=Exp_4157_train
#SBATCH --output=experiments/Exp_4157/output.log
#SBATCH --error=experiments/Exp_4157/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=46:00:00
#SBATCH --mem=22G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 4157"

set -e
module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate
export OMP_NUM_THREADS=4
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

python app_v2.py -g 0 -e 4157 -d "/home/epigou/scratch/metrics/Exp"
