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

# from decision_transformer.run_odt import run_odt, format_data

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

    environment_config_fname = 'configs/environment_config.yaml'
    eval_config_fname = 'configs/evaluation_config.yaml'
    algorithm_config_fname = 'configs/algorithm_config.yaml'    

    c = load_config_file(environment_config_fname)
    env_c = c['environment_settings']
    c = load_config_file(eval_config_fname)
    eval_c = c['eval_config']
    c = load_config_file(algorithm_config_fname)
    algorithm_dm = c['algorithm_settings']['algorithm']
    agent_by_zone= c['algorithm_settings']['agent_by_zone']
    federated_c = c['federated_learning_settings']

    if algorithm_dm == "DQN":
        c = load_config_file('configs/neural_network_config.yaml')
        num_episodes = c['nn_hyperparameters']['num_episodes']
    elif algorithm_dm == 'CMA_optimizer':
        c = load_config_file('configs/cma_config.yaml')
        num_episodes = c['cma_parameters']['max_generations']

    action_dim = env_c['action_dim'] * env_c['num_of_chargers']
    
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
            environment = EnvironmentClass(environment_config_fname, chargers_seeds[area_idx], env_c['coords'][area_idx], devices[area_idx], dtype=torch.float32)

            environment_list.append(environment)
            ev_info.append(environment.get_ev_info())
        
        elapsed_time = time.time() - start_time
    
        
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        all_routes = [None for route in env_c['coords']]
        for index, (city_lat, city_long) in enumerate(env_c['coords']):
            array_org_angle = rng.random(env_c['num_of_cars'])*2*np.pi # generating a list of random angles 
            all_routes[index] = get_org_dest_coords((city_lat, city_long), env_c['radius'], array_org_angle)

        elapsed_time = time.time() - start_time

        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        chargers = np.zeros(shape=[len(all_routes), env_c['num_of_cars'], env_c['num_of_chargers'] * 3, 3])
        
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
            if algorithm_dm == 'ODT':
                c = load_config_file('configs/neural_network_config.yaml')
                nn_c = c['odt_hyperparameters']               
                print(f"Training using ODT - Seed {seed}")
                chargers_copy = copy.deepcopy(chargers)
                run_odt(devices,
                          environment_list[nn_c['zone_index']],
                          chargers_copy,
                          all_routes[0],
                          action_dim,
                          eval_c['fixed_attributes'],
                          nn_c
                         )
                return

            
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f"Training using {algorithm_dm} - Seed {seed}", file=file)

            print(f"Training using {algorithm_dm} - Seed {seed}")


            metrics = []  # Used to track all metrics
            rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
            output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
            trajectories = []
            global_weights = None

            for aggregate_step in range(federated_c['aggregation_count']):

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
                    process = mp.Process(target=train_route, args=(charger_list, environment_list[ind],\
                                        all_routes[ind], date, action_dim, global_weights, aggregate_step,\
                                        ind, algorithm_dm, chargers_seeds[ind], seed, process_trajectories, eval_c['fixed_attributes'],\
                                        local_weights_list, process_rewards, process_metrics, process_output_values,\
                                        barrier, devices[ind], eval_c['verbose'], eval_c['display_training_times'],\
                                        agent_by_zone, eval_c['save_offline_data'], True))
                    processes.append(process)
                    process.start()

                print("Join Processes")

                for process in processes:
                    process.join()

                rewards = []
                for metric in process_metrics:
                    metric = metric[0]
                    to_print = f"Zone {metric['zone']+1} reward proccess { metric['rewards'][-1]:.3f}"+\
                        f" for aggregation: {metric['aggregation']+1}"
                    print(to_print)
                    with open(f'logs/{date}-training_logs.txt', 'a') as file:
                        print(to_print, file=file)
                        
                print("Join Weights")

                # Aggregate the weights from all local models
                global_weights = get_global_weights(local_weights_list, ev_info, federated_c['city_multiplier'],\
                                                    federated_c['zone_multiplier'], federated_c['model_multiplier'],\
                                                    agent_by_zone)

                # Save the global weights
                torch.save(global_weights, f'saved_networks/global_weights_{seed}.pth')

                # Extend the main lists with the contents of the process lists
                sorted_list = sorted([val[0] for sublist in process_rewards for val in sublist])
                print(f'Min and Max rewards for the aggregation step: {sorted_list[0],sorted_list[-1]}')
                rewards.extend(process_rewards)
                output_values.extend(process_output_values)
                metrics.extend(process_metrics)
                trajectories.extend(process_trajectories)

                with open(f'logs/{date}-training_logs.txt', 'a') as file:
                    print(f"\n\n############ Aggregation {aggregate_step + 1}/{federated_c['aggregation_count']} ############\n\n", file=file)

                print(f"\n\n############ Aggregation {aggregate_step + 1}/{federated_c['aggregation_count']} ############\n\n",)

            # Plot the aggregated data
            if eval_c['save_aggregate_rewards']:
                save_to_csv(rewards, 'outputs/rewards.csv')
                save_to_csv(output_values, 'outputs/output_values.csv')

                loaded_rewards = load_from_csv('outputs/rewards.csv')
                loaded_output_values = load_from_csv('outputs/output_values.csv')

                plot_aggregate_reward_data(loaded_rewards)
                plot_aggregate_output_values_per_route(loaded_output_values)

        else:
            metrics = []  # Used to track all metrics
            rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
            output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
            trajectories = []
            
            print(f"Loading saved models - Seed {seed}")
            global_weights = torch.load(f'saved_networks/global_weights_{seed}.pth')

            manager = mp.Manager()
            local_weights_list = manager.list([None for _ in range(len(chargers))])
            process_rewards = manager.list()
            process_output_values = manager.list()
            process_metrics = manager.list()
            process_trajectories = manager.list()

            # Barrier for synchronization
            barrier = mp.Barrier(len(chargers))

            processes = []
            for ind, charger_list in enumerate(chargers):
                process = mp.Process(target=train_route, args=(charger_list, environment_list[ind],\
                                    all_routes[ind], date, action_dim, global_weights, 0,\
                                    ind, algorithm_dm, chargers_seeds[ind], seed, process_trajectories, eval_c['fixed_attributes'],\
                                    local_weights_list, process_rewards, process_metrics, process_output_values,\
                                    barrier, devices[ind], eval_c['verbose'], eval_c['display_training_times'],\
                                    agent_by_zone, eval_c['save_offline_data'], False))
                processes.append(process)
                process.start()

            print("Join Processes")

            for process in processes:
                process.join()

            rewards = []
            for metric in process_metrics:
                metric = metric[0]
                to_print = f"Zone {metric['zone']+1} reward proccess { metric['rewards'][-1]:.3f}"+\
                    f" for aggregation: {metric['aggregation']+1}"
                print(to_print)
                with open(f'logs/{date}-training_logs.txt', 'a') as file:
                    print(to_print, file=file)

            # Extend the main lists with the contents of the process lists
            sorted_list = sorted([val[0] for sublist in process_rewards for val in sublist])
            print(f'Min and Max rewards for the aggregation step: {sorted_list[0],sorted_list[-1]}')
            rewards.extend(process_rewards)
            output_values.extend(process_output_values)
            metrics.extend(process_metrics)
            trajectories.extend(process_trajectories)

            if eval_c['fixed_attributes'] != [0, 1] and eval_c['fixed_attributes'] != [1, 0] and eval_c['fixed_attributes'] != [0.5, 0.5]:
                attr_label = 'learned'
            else:
                fixed_attributes = eval_c['fixed_attributes']
                attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

            if not os.path.exists('metrics/eval'):
                os.makedirs('metrics/eval')

            # Save all metrics from evaluation into a file
            evaluate(ev_info, metrics, seed, date, eval_c['verbose'], 'save', num_episodes, f"metrics/eval/metrics_{env_c['num_of_cars']}_{num_episodes}_{seed}_{attr_label}")

            # Generate the plots for the various metrics
            if eval_c['generate_plots']:
                evaluate(ev_info, None, seed, date, eval_c['verbose'], 'display', num_episodes, f"metrics/eval/metrics_{env_c['num_of_cars']}_{num_episodes}_{seed}_{attr_label}")


        if eval_c['fixed_attributes'] != [0, 1] and eval_c['fixed_attributes'] != [1, 0] and eval_c['fixed_attributes'] != [0.5, 0.5]:
            attr_label = 'learned'
        else:
            fixed_attributes = eval_c['fixed_attributes']
            attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

        if not os.path.exists('metrics/train'):
            os.makedirs('metrics/train')

        # Save all metrics from training into a file
        if eval_c['save_data'] and eval_c['train_model']:
            evaluate(ev_info, metrics, seed, date, eval_c['verbose'], 'save', num_episodes, f"metrics/train/metrics_{env_c['num_of_cars']}_{num_episodes}_{seed}_{attr_label}")

        # Generate the plots for the various metrics
        if eval_c['generate_plots']:
            evaluate(ev_info, None, seed, date, eval_c['verbose'], 'display', num_episodes, f"metrics/train/metrics_{env_c['num_of_cars']}_{num_episodes}_{seed}_{attr_label}")

        # if num_episodes != 1 and eval_c['continue_training']:
        #     user_input = input("More Episodes? ")
        # else:
        #     user_input = 'Done'

        et = time.time() - start_time
        to_print = f"Total time elapsed for this run"+\
            f"- et {str(int(et // 3600)).zfill(2)}:{str(int(et // 60) % 60).zfill(2)}:{str(int(et % 60)).zfill(2)}"

        print(to_print)
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(to_print, file=file)

        # Save offline data to pkl file
        if eval_c['save_offline_data']:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_path = f"/storage_1/epigou_storage/datasets/{env_c['seeds']}-{env_c['num_of_cars']}-{env_c['num_of_chargers']}-{federated_c['aggregation_count']}-{num_episodes}-{current_time}.pkl"

            traj_format = format_data(trajectories)
            with open(dataset_path, 'wb') as f:
                pickle.dump(traj_format, f)
                print('Offline Dataset Saved')

def train_route(chargers, environment, routes, date, action_dim, global_weights,
                aggregate_step, ind, algorithm_dm, sub_seed, main_seed, trajectories, fixed_attributes, local_weights_list, rewards, metrics, output_values, barrier, devices,
                verbose, display_training_times, agent_by_zone, save_offline_data, train_model):

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
        num_of_cars (int): Number of agents (EVs) in the environment.
        num_of_chargers (int): Number of charging stations.
        fixed_attributes (list): List of fixed attributes for redefining weights in the graph.
        local_weights_list (list): List to store the local weights of each agent.
        rewards (list): List to store the average rewards for each episode.
        metrics (list): List to store the various metrics collected during a simulation
        output_values (list): List to store the average output values for each episode.
        barrier (multiprocessing.Barrier): Barrier for synchronizing multiprocessing tasks.
        verbose (bool): Flag to enable detailed logging.
        display_training_times (bool): Flag to display training times for different operations.
        agent_by_zone (bool): True if using one agent for each zone, and false if using a agent for each car
        train_model (bool): True if training the model, False if evaluating the model

    Returns:
        None
    """

    try:
        # Create a deep copy of the environment for this thread
        chargers_copy = copy.deepcopy(chargers)

        if algorithm_dm == 'DQN':
            from train_dqn import train_dqn as train
            
        elif algorithm_dm == 'CMA_optimizer':
            from train_cma import train_cma as train
        
        else:
            raise RuntimeError(f'model {algorithm_dm} algorithm not found.')

        local_weights_per_agent, avg_rewards, avg_output_values, training_metrics, trajectories_per =\
            train(chargers_copy, environment, routes, date, action_dim, global_weights, aggregate_step,\
                  ind, sub_seed, main_seed, devices, agent_by_zone, fixed_attributes, verbose,\
                  display_training_times, torch.float32, save_offline_data, train_model)

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

        if train_model:
            local_weights_list[ind] = local_weights_per_agent

        print(f"Thread {ind} waiting")

        if train_model:
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