#!/bin/bash
#SBATCH --job-name=Exp_306_train
#SBATCH --output=experiments/Exp_306/output.log
#SBATCH --error=experiments/Exp_306/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-node=4
#SBATCH --time=14:00:00
#SBATCH --mem=180G

echo "Starting training for experiment 306"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 1 2 3 -e 306 -d "/home/epigou/scratch/metrics/Exp" 
    