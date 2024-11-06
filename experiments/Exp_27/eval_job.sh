#!/bin/bash
#SBATCH --job-name=Exp_27_eval
#SBATCH --output=experiments/Exp_27/output.log
#SBATCH --error=experiments/Exp_27/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --time=13:19:59
#SBATCH --mem=24G

echo "Starting evaluation for experiment 27"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 27 -d "/home/hartman/scratch/metrics/Exp" -eval True
    