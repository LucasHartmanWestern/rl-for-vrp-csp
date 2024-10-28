import os
import yaml
import argparse

def find_experiments_using_algorithm(algorithm_name):
    for exp_dir in os.listdir("./"):
        if exp_dir.startswith("Exp_"):
            exp_path = os.path.join("./", exp_dir)
            config_file = os.path.join(exp_path, "config.yaml")
            if os.path.isfile(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config.get("algorithm_settings", {}).get("algorithm") == algorithm_name:
                        print(f"Experiment {exp_dir} uses the algorithm {algorithm_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find experiments using a specific algorithm.")
    parser.add_argument("-a", "--algorithm", type=str, default="DQN", help="Name of the algorithm to search for")
    args = parser.parse_args()

    find_experiments_using_algorithm(args.algorithm)
