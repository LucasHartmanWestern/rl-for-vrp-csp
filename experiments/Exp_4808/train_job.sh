#!/bin/bash
#SBATCH --job-name=Exp_4808_train
#SBATCH --output=experiments/Exp_4808/output.log
#SBATCH --error=experiments/Exp_4808/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:10:00
#SBATCH --mem=45G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiments 4808 to 4810"

set -e

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate


mkdir -p $HOME/tmp
export OMP_NUM_THREADS=3

# Enable Nvidia MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Run 3 experiments in parallel on the same GPU
START_EXP=4808
NUM_EXPS=3

for ((i=START_EXP; i<NUM_EXPS+START_EXP; i++))
do
    echo "Launching experiment $i"
    python app_v2.py -g 0 -e "$i" -d "/home/epigou/scratch/metrics/Exp_$i" &
done

wait  # Wait for all background processes to finish
