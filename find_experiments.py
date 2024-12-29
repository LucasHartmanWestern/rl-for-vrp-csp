import argparse
import yaml

def check_match(exp_num, num_aggs, reward_type):
    # Load the config file
    with open(f'experiments/Exp_{exp_num}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Check if the number of aggregations matches
    if reward_type == 'Communal' and config['nn_hyperparameters']['average_rewards_when_training']:
        return config['federated_learning_settings']['aggregation_count'] == num_aggs
    elif reward_type == 'Greedy' and not config['nn_hyperparameters']['average_rewards_when_training']:
        return config['federated_learning_settings']['aggregation_count'] != num_aggs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-a','--num_aggs', type=int, default=None, help ='Target number of aggregations.')
    parser.add_argument('-r','--reward_type', type=str, default=None, help ='Communal or Greedy.')
    parser.add_argument('-e','--experiment_range', nargs='*', type=int, default=[], help ='Target experiment range.')
    args = parser.parse_args()

    for exp in range(*args.experiment_range):
        if check_match(exp, args.num_aggs, args.reward_type):
            print(f"Experiment {exp}")
