#!/bin/bash
#SBATCH --job-name=Exp_306
#SBATCH --output=experiments/Exp_306/output.log
#SBATCH --error=experiments/Exp_306/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=64G

echo "Starting training for experiment 306"

module load python/3.10 cuda cudnn
source ~/envs/merl3.9/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g0 -e 306 -d "/home/epigou/scratch/metrics/Exp"
    