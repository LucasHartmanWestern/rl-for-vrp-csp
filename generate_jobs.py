# Run this script to generate the job configuration files for the experiments

import argparse
import yaml

    
def create_job(args):    
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
        num_generations = config['cma_parameters']['max_generations']
        
    
        # Get the algorithm
        algorithm = config['algorithm_settings']['algorithm']
    
        # Calculate the time based on the total number of episodes
        # Note: these are rough estimates based on how long takes to train 10k episodes
        algorithm_time_mapping = {
            'DQN': 15 / 10000,
            'PPO': 40 / 10000,
            'CMA': 10 / 100
        }
    
        if algorithm in algorithm_time_mapping:
            num_cpus = num_gpus * 2
            mem_size = "64G"
            total_episodes = num_episodes * num_aggregations
            if algorithm == 'CMA':
                total_episodes = num_generations*num_aggregations
                mem_size = "24G"
                num_gpus /= 2 #on CMA two zones per gpu but 4 cpus per gpu
                num_cpus = num_gpus * 4
            calculated_time = algorithm_time_mapping[algorithm] * total_episodes
        else:
            print(f"Algorithm {algorithm} not supported. Need to add estimated duration for this algorithm.")
            continue
    
        data_dir = f"/home/{args.u}/scratch/metrics/Exp"
    
        # Generate the job config files
        job_script_content = f"""#!/bin/bash
    #SBATCH --job-name=Exp_{experiment}
    #SBATCH --output=experiments/Exp_{experiment}/output.log
    #SBATCH --error=experiments/Exp_{experiment}/error.log
    #SBATCH -A rrg-kgroling
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={num_cpus}
    #SBATCH --gpus-per-node={num_gpus}
    #SBATCH --time={str(int(calculated_time // 1)).zfill(2)}:{str(int((calculated_time * 60) % 60)).zfill(2)}:{str(int((calculated_time * 3600) % 60)).zfill(2)}
    #SBATCH --mem={mem_size}
    
    echo "Starting training for experiment {experiment}"
    
    module load python/3.10 cuda cudnn
    source ~/envs/merl_env/bin/activate
    
    # Enable multi-threading
    export OMP_NUM_THREADS={num_gpus * 2}
    
    python app_v2.py -g {" ".join(str(g) for g in range(num_gpus))} -e {experiment} -d "{data_dir}"
    """
    
        # Save job script to file
        with open(f'experiments/Exp_{experiment}/train_job.sh', 'w') as job_file:
            job_file.write(job_script_content)



if __name__ == "__main__":
    
    # Determine which experiments to create job config for
    parser = argparse.ArgumentParser(description="Generate job configuration files for experiments")
    parser.add_argument('-e', type=str, nargs='+', help="List of experiment numbers or 'all' to include all experiments")
    parser.add_argument('-u', type=str, nargs='', help="user account at DRAC")
    args = parser.parse_args()
    create_job(args)