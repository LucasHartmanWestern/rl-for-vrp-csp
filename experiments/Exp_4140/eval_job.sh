#!/bin/bash
#SBATCH --job-name=Exp_4140_eval
#SBATCH --output=experiments/Exp_4140/output.log
#SBATCH --error=experiments/Exp_4140/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=32:00:00
#SBATCH --mem=30G
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting evaluation for experiment 4140"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=4
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

python app_v2.py -g 0 -e 4140 -d "/home/epigou/scratch/metrics/Eval/Exp" -eval True
