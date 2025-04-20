#!/bin/bash
#SBATCH --job-name=Exp_4145_train
#SBATCH --output=experiments/Exp_4145/output.log
#SBATCH --error=experiments/Exp_4145/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
<<<<<<< HEAD
#SBATCH --time=30:00:00
#SBATCH --mem=15G
=======
#SBATCH --time=24:00:00
#SBATCH --mem=50G
>>>>>>> 26ca5fdef7e24bf203cb892634050949fbcd1603
#SBATCH --gpus-per-node=4

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=epigou@uwo.ca

echo "Starting training for experiment 4145, 4146, 4147"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4

# Activate Nvidia MPS:
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Run 3 experiments in parallel on the same GPU
START_EXP=4145
NUM_EXPS=3

for ((i=0; i<NUM_EXPS; i++))
do
    EXP_NUM=$((START_EXP + i))
    echo "Launching experiment $EXP_NUM"
    python app_v2.py -g 0 -e "$EXP_NUM" -d "/home/epigou/scratch/metrics/Exp_$EXP_NUM" &
done

<<<<<<< HEAD
wait  # Wait for all background processes to finish
=======
wait  # Wait for all background processes to finish
>>>>>>> 26ca5fdef7e24bf203cb892634050949fbcd1603
