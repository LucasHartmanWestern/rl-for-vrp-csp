from train_v2 import train
from data_loader import *
from visualize import *
import os
import argparse
import warnings
import time
import torch.multiprocessing as mp
from federated_learning import get_global_weights
import copy
from datetime import datetime
import numpy as np
from evaluation import evaluate
import pickle

from train_odt import train_odt

# from merl_env.env_class_v1_ import environment_class
from merl_env.environment import EnvironmentClass

mp.set_sharing_strategy('file_system')

mp.set_start_method('spawn', force=True)  # This needs to be done before you create any processes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_rl_vrp_csp(date, args):

    """
    Trains reinforcement learning models for vehicle routing and charging station placement (VRP-CSP).

    Parameters:
        date (str): The date string for logging purposes.

    Returns:
        None
    """

    ############ Initialization ############

    neural_network_config_fname = 'configs/neural_network_config.yaml'
    algorithm_config_fname = 'configs/algorithm_config.yaml'
    environment_config_fname = 'configs/environment_config.yaml'
    eval_config_fname = 'configs/evaluation_config.yaml'

    c = load_config_file(neural_network_config_fname)
    nn_c = c['nn_hyperparameters']
    c = load_config_file(algorithm_config_fname)
    algo_c = c['algorithm_settings']
    c = load_config_file(environment_config_fname)
    env_c = c['environment_settings']
    c = load_config_file(eval_config_fname)
    eval_c = c['eval_config']

    batch_size = int(nn_c['batch_size'])
    buffer_limit = int(nn_c['buffer_limit'])

    action_dim = nn_c['action_dim'] * env_c['num_of_chargers']
    
    #initializing GPUs for training
    n_gpus = len(args.list_gpus)
    n_zones = len(env_c['coords'])

    if n_gpus == 0:
        gpus = ['cpu'] # no gpus assigned, then everything running on cpu
        print(f'Woring with CPUs for all zones, with following configuration for zones and devices:')
    elif n_gpus == 1:
        gpus = [f'cuda:{args.list_gpus[0]}']
        print(f'Woring with one GPU -> {gpus[0]} for all zones, with following configuration for zones and devices:')

    elif (n_gpus > 1) & (n_gpus < torch.cuda.device_count()+1):
        gpus = [f'cuda:{args.list_gpus[i]}' for i in range(n_gpus)]
        print(f'Woring with {n_gpus} GPUs, with following configuration for zones and devices:')
    else:
        raise RuntimeError('Number of GPUs requested higher than available GPUs at server.')

    # Assign GPUs to zones in a round-robin fashion
    gpus_size = len(gpus)
    devices = [gpus[i % gpus_size] for i in range(n_zones)]
    
    for i, gpu in enumerate(devices):
        print(f'Zone {i} with {gpu}')

    
    # Run and train agents with different routes with reproducibility based on the selected seed
    for seed in env_c['seeds']:

        print(f'Running experiments with seed -> {seed}')
        # Creating and seeding a random generaton from Numpy
        rng = np.random.default_rng(seed)
        # Generating sub seeds to run on each environment
        chargers_seeds = rng.integers(low=0, high=10000, size=len(env_c['coords']))

        # Initializing list of enviroments
        environment_list = []
        ev_info = []
        start_time = time.time()
        for area_idx in range(n_zones):
            environment = EnvironmentClass(environment_config_fname, chargers_seeds[area_idx], devices[area_idx], dtype=torch.float32)
            environment_list.append(environment)
            ev_info.append(environment.get_ev_info())
        
        elapsed_time = time.time() - start_time
        
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        all_routes = [None for route in env_c['coords']]
        for index, (city_lat, city_long) in enumerate(env_c['coords']):
            array_org_angle = rng.random(env_c['num_of_agents'])*2*np.pi # generating a list of random angles 
            all_routes[index] = get_org_dest_coords((city_lat, city_long), env_c['radius'], array_org_angle)

        elapsed_time = time.time() - start_time

        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        chargers = np.zeros(shape=[len(all_routes), env_c['num_of_agents'], env_c['num_of_chargers'] * 3, 3])
        
        for route_id,  route in enumerate(all_routes):
            for agent_id, (org_lat, org_long, dest_lat, dest_long) in enumerate(route):
                data = get_charger_data()
                charger_info = np.c_[data['latitude'].to_list(), data['longitude'].to_list()]
                charger_list = get_charger_list(charger_info, org_lat, org_long, dest_lat, dest_long, env_c['num_of_chargers'])
                chargers[route_id][agent_id] = charger_list

        elapsed_time = time.time() - start_time
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get Chargers: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get Chargers: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

            
        if eval_c['train_model']:
            if algo_c['algorithm'] == 'ODT':
                print(f"Training using ODT - Seed {seed}")
                chargers_copy = copy.deepcopy(chargers)
                train_odt(devices,
                          environment_list[0],
                          chargers_copy,
                          all_routes[0],
                          action_dim,
                          eval_c['fixed_attributes'],
                          algo_c
                         )
            return
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f"Training using Deep-Q Learning - Seed {seed}", file=file)

            print(f"Training using Deep-Q Learning - Seed {seed}")

            metrics = []  # Used to track all metrics
            rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
            output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
            trajectories = []
            global_weights = None

            for aggregate_step in range(nn_c['aggregation_count']):

                manager = mp.Manager()
                local_weights_list = manager.list([None for _ in range(len(chargers))])
                process_rewards = manager.list()
                process_output_values = manager.list()
                process_metrics = manager.list()
                process_trajectories = manager.list()

                # Barrier for synchronization
                barrier = mp.Barrier(len(chargers))

                # Creating output directory
                folder = 'outputs/best_paths/'
                if not os.path.exists(folder):
                    os.makedirs(folder)

                processes = []
                for ind, charger_list in enumerate(chargers):
                    process = mp.Process(target=train_route, args=(
                        charger_list, environment_list[ind], all_routes[ind], date, action_dim,
                        global_weights, aggregate_step, ind, chargers_seeds[ind], seed, nn_c['epsilon'],
                        nn_c['epsilon_decay'], nn_c['discount_factor'], nn_c['learning_rate'],
                        nn_c['num_episodes'], batch_size, buffer_limit, process_trajectories, nn_c['layers'],
                        eval_c['fixed_attributes'], local_weights_list, process_rewards, process_metrics,
                        process_output_values, barrier, devices[ind], eval_c['verbose'], 
                        eval_c['display_training_times'], nn_c['nn_by_zone'], eval_c['save_offline_data']))
                    processes.append(process)
                    process.start()

                print("Join Processes")

                for process in processes:
                    process.join()

                print("Join Weights")

                # Aggregate the weights from all local models
                global_weights = get_global_weights(local_weights_list, ev_info, nn_c['city_multiplier'], nn_c['zone_multiplier'], nn_c['model_multiplier'], nn_c['nn_by_zone'])

                # Extend the main lists with the contents of the process lists
                sorted_list = sorted([val[0] for sublist in process_rewards for val in sublist])
                print(f'Min and Max rewards for the aggregation step: {sorted_list[0],sorted_list[-1]}')
                rewards.extend(process_rewards)
                output_values.extend(process_output_values)
                metrics.extend(process_metrics)
                trajectories.extend(process_trajectories)

                with open(f'logs/{date}-training_logs.txt', 'a') as file:
                    print(f"\n\n############ Aggregation {aggregate_step + 1}/{nn_c['aggregation_count']} ############\n\n", file=file)

                print(f"\n\n############ Aggregation {aggregate_step + 1}/{nn_c['aggregation_count']} ############\n\n",)

            # Plot the aggregated data
            if eval_c['save_aggregate_rewards']:
                save_to_csv(rewards, 'outputs/rewards.csv')
                save_to_csv(output_values, 'outputs/output_values.csv')

                loaded_rewards = load_from_csv('outputs/rewards.csv')
                loaded_output_values = load_from_csv('outputs/output_values.csv')

                plot_aggregate_reward_data(loaded_rewards)
                plot_aggregate_output_values_per_route(loaded_output_values)



        if eval_c['fixed_attributes'] != [0, 1] and eval_c['fixed_attributes'] != [1, 0] and eval_c['fixed_attributes'] != [0.5, 0.5]:
            attr_label = 'learned'
        else:
            fixed_attributes = eval_c['fixed_attributes']
            attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

        # Save all metrics from training into a file
        if eval_c['save_data'] and eval_c['train_model']:
            evaluate(ev_info, metrics, seed, date, eval_c['verbose'], 'save', nn_c['num_episodes'], f"metrics/metrics_{env_c['num_of_agents']}_{nn_c['num_episodes']}_{seed}_{attr_label}")

        # Generate the plots for the various metrics
        if eval_c['generate_plots']:
            evaluate(ev_info, None, seed, date, eval_c['verbose'], 'display', nn_c['num_episodes'], f"metrics/metrics_{env_c['num_of_agents']}_{nn_c['num_episodes']}_{seed}_{attr_label}")

        if nn_c['num_episodes'] != 1 and eval_c['continue_training']:
            user_input = input("More Episodes? ")
        else:
            user_input = 'Done'

        # Save offline data to pkl file
        if eval_c['save_offline_data']:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_path = f'data/offline-data_{current_time}.pkl'
            with open(dataset_path, 'wb') as f:
                pickle.dump(trajectories, f)
                print('Offline Dataset Saved')


def train_route(chargers, environment, routes, date, action_dim, global_weights,
                aggregate_step, ind, sub_seed, main_seed, epsilon, epsilon_decay,
                discount_factor, learning_rate, num_episodes, batch_size,
                buffer_limit, trajectories, layers, fixed_attributes,
                local_weights_list, rewards, metrics, output_values, barrier, devices,
                verbose, display_training_times, nn_by_zone, save_offline_data):

    """
    Trains a single route for the VRP-CSP problem using reinforcement learning in a multiprocessing environment.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        environment (class): Class containing information about the environment.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the Q-networks.
        aggregate_step (int): Aggregation step number for tracking.
        ind (int): Index of the current process.
        sub_seed (int): Sub-seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        epsilon (float): Initial exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Decay rate for the exploration rate.
        discount_factor (float): Discount factor for future rewards.
        learning_rate (float): Learning rate for the optimizer.
        num_episodes (int): Number of training episodes.
        batch_size (int): Size of the mini-batch for experience replay.
        buffer_limit (int): Maximum size of the experience replay buffer.
        num_of_agents (int): Number of agents (EVs) in the environment.
        num_of_chargers (int): Number of charging stations.
        layers (list): List of integers defining the architecture of the neural networks.
        fixed_attributes (list): List of fixed attributes for redefining weights in the graph.
        local_weights_list (list): List to store the local weights of each agent.
        rewards (list): List to store the average rewards for each episode.
        metrics (list): List to store the various metrics collected during a simulation
        output_values (list): List to store the average output values for each episode.
        barrier (multiprocessing.Barrier): Barrier for synchronizing multiprocessing tasks.
        verbose (bool): Flag to enable detailed logging.
        display_training_times (bool): Flag to display training times for different operations.
        nn_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car


    Returns:
        None
    """

    try:
        # Create a deep copy of the environment for this thread
        chargers_copy = copy.deepcopy(chargers)

        local_weights_per_agent, avg_rewards, avg_output_values, training_metrics, trajectories_per =\
            train(chargers_copy, environment, routes, date, action_dim, global_weights, aggregate_step,\
                  ind, sub_seed, main_seed, epsilon, epsilon_decay, discount_factor, learning_rate, \
                  num_episodes, batch_size, buffer_limit, layers, devices, fixed_attributes, verbose,\
                  display_training_times, torch.float32, nn_by_zone, save_offline_data)

        # Save results of training
        st = time.time()
        rewards.append(avg_rewards)
        output_values.append(avg_output_values)
        metrics.append(training_metrics)
        trajectories.append(trajectories_per)
        et = time.time() - st

        if verbose:
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f'Spent {et:.3f} seconds saving results', file=file)  # Print saving time with 3 decimal places
            print(f'Spent {et:.3f} seconds saving results')  # Print saving time with 3 decimal places

        local_weights_list[ind] = local_weights_per_agent

        print(f"Thread {ind} waiting")

        barrier.wait()  # Wait for all threads to finish before proceeding

    except Exception as e:
        print(f"Error in process {ind} during aggregate step {aggregate_step}: {str(e)}")
        raise


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-c','--number_processors', type=int, default=1,help='number of processors used to run MERL')
    parser.add_argument('-g','--list_gpus', nargs='*', type=int, default=[], help ='Request of enumerated gpus run MERL.')
    args = parser.parse_args()
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d_%H-%M')

    train_rl_vrp_csp(date, args)
