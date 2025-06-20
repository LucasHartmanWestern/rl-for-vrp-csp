#!/bin/bash
#SBATCH --job-name=Exp_53_eval
#SBATCH --output=experiments/Exp_0053/output.log
#SBATCH --error=experiments/Exp_0053/error.log
#SBATCH -A rrg-kgroling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0053:19:59
#SBATCH --mem=24G

echo "Starting evaluation for experiment 0053"

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py -g 0 -e 0053 -d "/home/hartman/scratch/metrics/Exp" -eval True
    