#!/bin/bash
#SBATCH --job-name=Exp_4810_train
#SBATCH --output=experiments/Exp_4810/output.log
#SBATCH --error=experiments/Exp_4810/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:10:00
#SBATCH --mem=15G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiments 4808 to 4810"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

export OMP_NUM_THREADS=3

# Enable Nvidia MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Run 3 experiments in parallel on the same GPU
START_EXP=4808
NUM_EXPS=3

for ((i=0; i<NUM_EXPS; i++))
do
    EXP_NUM=$((START_EXP + i))
    echo "Launching experiment $EXP_NUM"
    python app_v2.py -g 0 -e "$EXP_NUM" -d "/home/epigou/scratch/metrics/Exp_$EXP_NUM" &
done

wait  # Wait for all background processes to finish
