import subprocess
import argparse
import yaml
import os

def run_exp_on_drac(start_experiment, end_experiment, algorithm=None, eval=False, seed=None, aggregation=None):
    """
    Run experiments using the servers from the Digital Research Alliance of Canada (DRAC)

    Parameters:
        start_experiment (int): Start experiment number
        end_experiment (int): End experiment number
        algorithm (str): Algorithm to run
        eval (bool): Whether to evaluate the model
        seed (int): Seed to run
        aggregation (int): Aggregation count to run
    """

    for experiment_number in range(start_experiment, end_experiment + 1):
        try:
            # Load config.yaml file for experiment
            config_file = os.path.join(f"../experiments/Exp_{experiment_number}", "config.yaml")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

                if algorithm: # Filter by algorithm
                    if config.get("algorithm_settings", {}).get("algorithm") != algorithm:
                        print(f"Experiment {experiment_number} does not use algorithm {algorithm}")
                        continue

                if seed: # Filter by seed
                    if int(config.get("environment_settings", {}).get("seed")) != int(seed):
                        print(f"Experiment {experiment_number} does not use seed {seed}")
                        continue

                if aggregation: # Filter by aggregation count
                    if int(config.get("federated_learning_settings", {}).get("aggregation_count")) != int(aggregation):
                        print(f"Experiment {experiment_number} does not use aggregation {aggregation}")
                        continue

        except Exception as e:
            print(f"Error loading config.yaml file for experiment {experiment_number}: {e}")
            continue

        cmd = f"sbatch experiments/Exp_{experiment_number}/{'eval' if eval else 'train'}_job.sh"
        print(f"Running command: {cmd}")
        try:
            subprocess.run(cmd, shell=True)
        except Exception as e:
            print(f"Error running command: {cmd}")
            print(f"Error message: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-e','--experiments_list', nargs='*', type=int, default=[], help ='Get the list of experiment to run.')
    parser.add_argument('-a', '--algorithm', type=str, help='Algorithm to run.')
    parser.add_argument('-s', '--seed', type=str, default=None, help='Seed to run.')
    parser.add_argument('-agg', '--aggregation', type=str, default=None, help='Aggregation count to run.')
    parser.add_argument('-eval', type=bool, default=False, help="Evaluate the model")
    args = parser.parse_args()

    start_experiment = args.experiments_list[0]
    end_experiment = args.experiments_list[-1]

    if args.algorithm:
        print(f"Running experiments in range {start_experiment} to {end_experiment} using ONLY algorithm {args.algorithm}")
    else:
        print(f"Running experiments from {start_experiment} to {end_experiment}")

    run_exp_on_drac(start_experiment, end_experiment, args.algorithm, args.eval, args.seed, args.aggregation)
