import subprocess
import argparse
import yaml
import os

def run_exp_on_drac(start_experiment, end_experiment, algorithm=None):
    for experiment_number in range(start_experiment, end_experiment + 1):
        try:
            if algorithm:
                # Load config.yaml file for experiment
                config_file = os.path.join(f"./experiments/Exp_{experiment_number}", "config.yaml")
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                    # Check if the experiment uses the specified algorithm
                    if config.get("algorithm_settings", {}).get("algorithm") != algorithm:
                        print(f"Experiment {experiment_number} does not use algorithm {algorithm}")
                        continue
        except Exception as e:
            print(f"Error loading config.yaml file for experiment {experiment_number}: {e}")
            continue

        cmd = f"sbatch experiments/Exp_{experiment_number}/train_job.sh"
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
    args = parser.parse_args()

    start_experiment = args.experiments_list[0]
    end_experiment = args.experiments_list[-1]

    if args.algorithm:
        print(f"Running experiments in range {start_experiment} to {end_experiment} using ONLY algorithm {args.algorithm}")
    else:
        print(f"Running experiments from {start_experiment} to {end_experiment}")

    run_exp_on_drac(start_experiment, end_experiment, args.algorithm)
