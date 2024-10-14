#!/bin/bash
#SBATCH --job-name=Exp_0
#SBATCH --output=experiments/Exp_0/output.log
#SBATCH --error=experiments/Exp_0/error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --time=00:07:12
#SBATCH --mem=32G

echo "Starting training for experiment 0"
nvidia-smi

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=4

python app_v2.py -g 0 1 -e 0
