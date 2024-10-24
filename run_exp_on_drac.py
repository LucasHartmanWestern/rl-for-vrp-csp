import subprocess
import argparse

def run_exp_on_drac(start_experiment, end_experiment):
    for experiment_number in range(start_experiment, end_experiment + 1):
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
    args = parser.parse_args()

    start_experiment = args.experiments_list[0]
    end_experiment = args.experiments_list[-1]

    print(f"Running experiments from {start_experiment} to {end_experiment}")

    run_exp_on_drac(start_experiment, end_experiment)
