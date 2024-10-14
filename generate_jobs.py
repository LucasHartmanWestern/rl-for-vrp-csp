# Run this script to generate the job configuration files for the experiments

import argparse
import yaml

if __name__ == "__main__":
    
    # Determine which experiments to create job config for
    parser = argparse.ArgumentParser(description="Generate job configuration files for experiments")
    parser.add_argument('-e', type=str, nargs='+', help="List of experiment numbers or 'all' to include all experiments")
    args = parser.parse_args()

    experiment_list = []
    if args.e:
        if 'all' in args.e:
            import os
            experiment_list = [int(d.split('_')[1]) for d in os.listdir('experiments') if d.startswith('Exp_')]
        else:
            experiment_list = [int(e) for e in args.e]
    
    print("Experiments to create job config for:", experiment_list)

# Read config file for each experiment
for experiment in experiment_list:
    with open(f'experiments/Exp_{experiment}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Use 1 gpu per zone
    num_gpus = len(config['environment_settings']['coords'])
    
    # Get the number of episodes and aggregations
    num_episodes = config['nn_hyperparameters']['num_episodes']
    num_aggregations = config['federated_learning_settings']['aggregation_count']
    total_episodes = num_episodes * num_aggregations

    # Get the algorithm
    algorithm = config['algorithm_settings']['algorithm']

    # Calculate the time based on the total number of episodes
    # Note: these are rough estimates based on how long takes to train 10k episodes
    algorithm_time_mapping = {
        'DQN': 24,
        'PPO': 55
    }

    if algorithm in algorithm_time_mapping:
        calculated_time = algorithm_time_mapping[algorithm] * (total_episodes / 10000)
    else:
        raise Exception(f"Algorithm {algorithm} not supported. Need to add estimated duration for this algorithm.")

    # Generate the job config files
    job_script_content = f"""#!/bin/bash
#SBATCH --job-name=Exp_{experiment}
#SBATCH --output=experiments/Exp_{experiment}/output.log
#SBATCH --error=experiments/Exp_{experiment}/error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time={str(int(calculated_time // 1)).zfill(2)}:{str(int((calculated_time * 60) % 60)).zfill(2)}:{str(int((calculated_time * 3600) % 60)).zfill(2)}
#SBATCH --mem=32G

module load python/3.10 cuda cudnn
source ~/envs/merl_env/bin/activate

python app_v2.py -g {" ".join(str(g) for g in range(num_gpus))} -e {experiment}
"""

    # Save job script to file
    with open(f'experiments/Exp_{experiment}/train_job.sh', 'w') as job_file:
        job_file.write(job_script_content)