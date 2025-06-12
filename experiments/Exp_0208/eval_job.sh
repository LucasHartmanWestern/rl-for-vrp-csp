#!/bin/bash
#SBATCH --job-name=Exp_208_eval
#SBATCH --output=experiments/Exp_0208/output.log
#SBATCH --error=experiments/Exp_0208/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=05:39:59
#SBATCH --mem=64G

echo "Starting evaluation for experiment 0208"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 0208 -d "/home/hartman/scratch/metrics/Exp" -eval True
    