#!/bin/bash
#SBATCH --job-name=Exp_4057_train
#SBATCH --output=experiments/Exp_4057/output.log
#SBATCH --error=experiments/Exp_4057/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=133:00:00
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment 4057"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 4057 -d "/home/hartman/scratch/metrics/Exp" 
    