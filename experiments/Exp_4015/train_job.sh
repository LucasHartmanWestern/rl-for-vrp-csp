#!/bin/bash
#SBATCH --job-name=Exp_4015_train
#SBATCH --output=experiments/Exp_4015/output.log
#SBATCH --error=experiments/Exp_4015/error.log
#SBATCH -A def-mcapretz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=90:00:00
#SBATCH --mem=64G


#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=lhartma8@uwo.ca

echo "Starting training for experiment 4015"

set -e  # Exit immediately if a command exits with a non-zero status

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

# Enable multi-threading
export OMP_NUM_THREADS=2

python app_v2.py  -e 4015 -d "/home/hartman/scratch/metrics/Exp" 
    