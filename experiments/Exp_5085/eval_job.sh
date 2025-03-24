#!/bin/bash
#SBATCH --job-name=Exp_5085_eval
#SBATCH --output=experiments/Exp_5085/output.log
#SBATCH --error=experiments/Exp_5085/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=16G


#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting evaluation for experiment 5085"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 5085 -d "/home/sgomezro/scratch/metrics/Exp" -eval True
    