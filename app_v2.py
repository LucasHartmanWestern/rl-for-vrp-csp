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
from codecarbon import EmissionsTracker
import uuid
import shutil
import pandas as pd

from collections import defaultdict

warnings.filterwarnings("ignore")


# from merl_env.env_class_v1_ import environment_class
from merl_env.environment import EnvironmentClass

mp.set_sharing_strategy('file_system')

mp.set_start_method('spawn', force=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_rl_vrp_csp(args):

    """
    Trains reinforcement learning models for vehicle routing and charging station placement (VRP-CSP).

    Parameters:
        args ()

    Returns:
        None
    """

    ############ Initialization ############
    #get current date for experiments
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d_%H-%M')

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")

    # Get the list of available GPUs from the environment variable
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        # Split the string into a list of GPU indices
        available_gpus = cuda_visible_devices.split(',')
    else:
        # If the environment variable is not set, use all available GPUs
        available_gpus = [str(i) for i in range(torch.cuda.device_count())]

    #initializing GPUs for training
    n_gpus = len(args.list_gpus)
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

    # Verify that the GPUs requested by the user are available
    for gpu in gpus:
        gpu_index = gpu.replace('cuda:', '')
        if gpu_index not in available_gpus and gpu != 'cpu':
            raise RuntimeError(f"Requested GPU {gpu_index} is not available.")

    data_dir = args.data_dir

    #Fire up initialization
    init_fname = 'experiments/initial_config.yaml'
    init_config = load_config_file(init_fname)
    
    #Getting experiments list to run
    experiment_list = args.experiments_list
    if len(experiment_list) == 0: 
        # No experiment given in cosole, then getting initial configuration experiments list
        experiment_list = init_config['experiment_list']
    
    #Getting into Training or Evaluating mode to run experiments
    run_mode = init_config['model_run_mode']
    # Check if run_mode is either "Training" or "Testing"
    if run_mode not in ["Training", "Testing"]:
        raise ValueError(f"Invalid run_mode: '{run_mode}'. Expected 'Training' or 'Testing'.")

    #Should continue the training
    load_existing_model = init_config['continue_training']

    # Make logs directory if it doesn't already exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Run each experiment on experiments list
    for experiment_number in experiment_list:

        config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
        c = load_config_file(config_fname)
        env_c = c['environment_settings']
        eval_c = c['eval_config']
        algorithm_dm = c['algorithm_settings']['algorithm']
        agent_by_zone= c['algorithm_settings']['agent_by_zone']
        federated_c = c['federated_learning_settings']

        if algorithm_dm in ["DQN", "PPO", "DDPG"]:
            num_episodes = c['nn_hyperparameters']['num_episodes']
        elif algorithm_dm == 'CMA':
            num_episodes = c['cma_parameters']['max_generations']
        if algorithm_dm == 'ODT':
            variant = c['odt_hyperparameters']
        else:
            variant = None

        action_dim = env_c['action_dim'] * env_c['num_of_chargers']
        #saving metric resutls from experiments
        metrics_base_path = f"{data_dir}_{experiment_number}" if data_dir else f"{c['eval_config']['save_path_metrics']}_{experiment_number}"

        print(f"Saving metrics to base path: {metrics_base_path}")

        if os.path.exists(f'{metrics_base_path}/train') and run_mode == "Training":
            shutil.rmtree(f'{metrics_base_path}/train')

        # Now assign GPUs to zones
        n_zones = len(env_c['coords'])
        gpus_size = len(gpus)
        exp_devices = [gpus[i % gpus_size] for i in range(n_zones)]
        if n_gpus == 0:
            devices = ['cpu' for _ in range(n_zones)]
        else:
            for i, gpu in enumerate(exp_devices):
                print(f'Zone {i} with GPU {gpu} - {torch.cuda.get_device_name(gpu)}')

        devices = [gpus[i % gpus_size] for i in range(n_zones)]

        # get seed for current experiment
        seed = env_c['seed']
        # Run and train agents with different routes with reproducibility based on the selected seed

        #Retrieve Training or Evaluation mode and Continue or from scrath model training
        if run_mode == "Testing":
            global_weights = torch.load(f'saved_networks/Exp_{experiment_number}/global_weights.pth')
        elif load_existing_model:
            global_weights = torch.load(f'saved_networks/Exp_{experiment_number}/global_weights.pth')
            

        #to ask Lucas
        if eval_c['evaluate_on_diff_seed'] or args.eval:
            seed_options = [1234, 5555, 2020]
            seed_index = seed_options.index(seed)
            old_seed = seed
            seed = seed_options[(seed_index + 1) % len(seed_options)]

            print(f'Running experiments with model trained on seed {old_seed} on new seed {seed}')

        else:
            print(f'Running experiment {experiment_number} in {run_mode} mode with seed -> {seed}')

        # Creating and seeding a random generator from Numpy
        rng = np.random.default_rng(seed)
        # Generating sub seeds to run on each environment
        chargers_seeds = rng.integers(low=0, high=10000, size=len(env_c['coords']))

        # Initializing list of enviroments
        environment_list = []
        ev_info = []
        start_time = time.time()
        for area_idx in range(n_zones):
            environment = EnvironmentClass(config_fname, seed, chargers_seeds[area_idx], env_c['coords'][area_idx], device=devices[area_idx], dtype=torch.float32)

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

        # Store carbon emissions data
        emission_output_dir = f"{metrics_base_path}/{'train' if run_mode == 'Training' else 'eval'}"
        if not os.path.exists(emission_output_dir):
            os.makedirs(emission_output_dir)

        if run_mode == "Training":
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f"Training using {algorithm_dm} - Seed {seed}", file=file)

            print(f"Training using {algorithm_dm} - Seed {seed}")


            metrics = []  # Used to track all metrics
            rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
            output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
            trajectories = []
            old_buffers = [None for _ in range(len(chargers))] # Hold the buffers for the previous aggregation step
            global_weights = None

            for aggregate_step in range(federated_c['aggregation_count']):
                try:
                    # Start tracking emissions
                    tracker = EmissionsTracker(
                        output_dir=emission_output_dir,
                        save_to_file=f"emissions.csv",  # Temporary file
                        tracking_mode='process',
                        log_level='error'
                    )
                    tracker.start()

                    manager = mp.Manager()
                    local_weights_list = manager.list([None for _ in range(len(chargers))])
                    process_rewards = manager.list()
                    process_output_values = manager.list()
                    process_metrics = manager.list()
                    process_trajectories = manager.list()
                    process_buffers = manager.list([None for _ in range(len(chargers))])

                    # Barrier for synchronization
                    barrier = mp.Barrier(len(chargers))

                    # Creating output directory
                    folder = 'outputs/best_paths/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    processes = []
                    for ind, charger_list in enumerate(chargers):
                        process = mp.Process(target=train_route, args=(ev_info, metrics_base_path, experiment_number, charger_list, environment_list[ind],\
                                            all_routes[ind], date, action_dim, global_weights, aggregate_step,\
                                            ind, algorithm_dm, chargers_seeds[ind], seed, process_trajectories, args, eval_c['fixed_attributes'],\
                                            local_weights_list, process_rewards, process_metrics, process_output_values,\
                                            barrier, devices[ind], eval_c['verbose'], eval_c['display_training_times'],\
                                            agent_by_zone, variant, eval_c['save_offline_data'], True, old_buffers[ind], process_buffers))
                        processes.append(process)
                        process.start()

                    print("Join Processes")

                    for process in processes:
                        process.join()

                    rewards = []
                    # for metric in process_metrics:
                    #     metric = metric[0]
                    #     to_print = f"Zone {metric['zone']+1} reward proccess { metric['rewards'][-1]:.3f}"+\
                    #         f" for aggregation: {metric['aggregation']+1}"
                    #     print(to_print)
                    #     with open(f'logs/{date}-training_logs.txt', 'a') as file:
                    #         print(to_print, file=file)
                            
                    print("Join Weights")

                    # Aggregate the weights from all local models
                    if algorithm_dm == 'ODT':
                        # Aggregate the weights from all local models
                        global_weights = get_global_weights(local_weights_list, ev_info, federated_c['city_multiplier'],\
                                                            federated_c['zone_multiplier'], federated_c['model_multiplier'],\
                                                            agent_by_zone, True)
                    else:
                                            # Aggregate the weights from all local models
                        global_weights = get_global_weights(local_weights_list, ev_info, federated_c['city_multiplier'],\
                                                            federated_c['zone_multiplier'], federated_c['model_multiplier'],\
                                                            agent_by_zone)

                    save_global_path = f'saved_networks/Exp_{experiment_number}/'
                    if not os.path.exists(save_global_path):
                        os.makedirs(save_global_path)
                    # Save the global weights
                    torch.save(global_weights, f'{save_global_path}/global_weights.pth')

                    # Extend the main lists with the contents of the process lists
                    sorted_list = sorted([val[0] for sublist in process_rewards for val in sublist])
                    if sorted_list:
                        print(f'Min and Max rewards for the aggregation step: {sorted_list[0], sorted_list[-1]}')
                    else:
                        print("No rewards found for this aggregation step.")
                    rewards.extend(process_rewards)
                    output_values.extend(process_output_values)
                    metrics.extend(process_metrics)
                    trajectories.extend(process_trajectories)
                    old_buffers = list(process_buffers)

                    with open(f'logs/{date}-training_logs.txt', 'a') as file:
                        print(f"\n\n############ Aggregation {aggregate_step + 1}/{federated_c['aggregation_count']} ############\n\n", file=file)

                    print(f"\n\n############ Aggregation {aggregate_step + 1}/{federated_c['aggregation_count']} ############\n\n",)

                finally:
                    # Stop tracking emissions
                    emissions = tracker.stop()
                    print(f"Total CO₂ emissions: {emissions} kg")
                    try:
                        # Read the temporary emissions report
                        temp_df = pd.read_csv(f"{emission_output_dir}/emissions.csv")

                        # Remove the temporary emissions file
                        os.remove(f"{emission_output_dir}/emissions.csv")

                        # Add the aggregate_step column
                        temp_df['aggregate_step'] = aggregate_step

                        # Determine the write mode based on the aggregate_step
                        write_mode = 'w' if aggregate_step == 0 else 'a'

                        # Write or append the updated DataFrame to the main CSV
                        with open(f"{emission_output_dir}/emissions_report.csv", write_mode) as f:
                            temp_df.to_csv(f, header=(write_mode == 'w'), index=False)
                    except Exception as e:
                        print(f"Error saving emissions report")

            # Plot the aggregated data
            if eval_c['save_aggregate_rewards']:
                save_to_csv(rewards, 'outputs/rewards.csv')
                save_to_csv(output_values, 'outputs/output_values.csv')

                loaded_rewards = load_from_csv('outputs/rewards.csv')
                loaded_output_values = load_from_csv('outputs/output_values.csv')

                plot_aggregate_reward_data(loaded_rewards)
                plot_aggregate_output_values_per_route(loaded_output_values)

        elif run_mode == "Testing":
            metrics = []  # Used to track all metrics
            rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
            output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
            trajectories = []
            old_buffers = [None for _ in range(len(chargers))] # Hold the buffers for the previous aggregation step
            
            print(f"Loading saved models - Seed {seed}")
            global_weights = torch.load(f'saved_networks/Exp_{experiment_number}/global_weights.pth')

            manager = mp.Manager()
            local_weights_list = manager.list([None for _ in range(len(chargers))])
            process_rewards = manager.list()
            process_output_values = manager.list()
            process_metrics = manager.list()
            process_trajectories = manager.list()
            process_buffers = manager.list([None for _ in range(len(chargers))])

            # Barrier for synchronization
            barrier = mp.Barrier(len(chargers))

            processes = []
            for ind, charger_list in enumerate(chargers):
                process = mp.Process(target=train_route, args=(ev_info, metrics_base_path, experiment_number, charger_list, environment_list[ind],\
                                    all_routes[ind], date, action_dim, global_weights, 0,\
                                    ind, algorithm_dm, chargers_seeds[ind], seed, process_trajectories, args, eval_c['fixed_attributes'],\
                                    local_weights_list, process_rewards, process_metrics, process_output_values,\
                                    barrier, devices[ind], eval_c['verbose'], eval_c['display_training_times'],\
                                    agent_by_zone, variant, eval_c['save_offline_data'], False, old_buffers[ind], process_buffers))
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
            old_buffers = list(process_buffers)

            if eval_c['fixed_attributes'] != [0, 1] and eval_c['fixed_attributes'] != [1, 0] and eval_c['fixed_attributes'] != [0.5, 0.5]:
                attr_label = 'learned'
            else:
                fixed_attributes = eval_c['fixed_attributes']
                attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

            if not os.path.exists(f'{metrics_base_path}/eval'):
                os.makedirs(f'{metrics_base_path}/eval')
            # Save all metrics from evaluation into a file
            evaluate(ev_info, metrics, seed, date, eval_c['verbose'], 'save', num_episodes, f"{metrics_base_path}/eval/metrics")

            # Generate the plots for the various metrics
            if eval_c['generate_plots']:
                evaluate(ev_info, None, seed, date, eval_c['verbose'], 'display', num_episodes, f"{metrics_base_path}/eval/metrics")

        if eval_c['fixed_attributes'] != [0, 1] and eval_c['fixed_attributes'] != [1, 0] and eval_c['fixed_attributes'] != [0.5, 0.5]:
            attr_label = 'learned'
        else:
            fixed_attributes = eval_c['fixed_attributes']
            attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

        print(f'directory {metrics_base_path}')
        if not os.path.exists(f'{metrics_base_path}/train'):
            os.makedirs(f'{metrics_base_path}/train')

        # # Save all metrics from training into a file
        # if eval_c['save_data']:
        #     evaluate(ev_info, metrics, seed, date, eval_c['verbose'], 'save', num_episodes, f"{metrics_base_path}/train/metrics")

        # # Generate the plots for the various metrics
        # if eval_c['generate_plots']:
        #     evaluate(ev_info, None, seed, date, eval_c['verbose'], 'display', num_episodes, f"{metrics_base_path}/train/metrics")

        et = time.time() - start_time
        to_print = f"Total time elapsed for this run"+\
            f"- et {str(int(et // 3600)).zfill(2)}:{str(int(et // 60) % 60).zfill(2)}:{str(int(et % 60)).zfill(2)}"

        print(to_print)
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(to_print, file=file)

        # Save offline data to pkl file
        if eval_c['save_offline_data']:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            dataset_path = f"{metrics_base_path}/data-{current_time}.pkl"
            print({dataset_path})

            traj_format = format_data(trajectories)
            with open(dataset_path, 'wb') as f:
                pickle.dump(traj_format, f)
                print('Offline Dataset Saved')

def train_route(ev_info, metrics_base_path, experiment_number, chargers, environment, routes, date, action_dim, global_weights,
                aggregate_step, ind, algorithm_dm, sub_seed, main_seed, trajectories, args, fixed_attributes, local_weights_list,
                rewards, metrics, output_values, barrier, device, verbose, display_training_times, agent_by_zone, variant,
                save_offline_data, train_model, old_buffers, process_buffers):

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
        args (argparse.Namespace): Command-line arguments.
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

        print(f'algorithm dm {algorithm_dm}')

        if algorithm_dm == 'DQN':
            from training_processes.train_dqn import train_dqn as train

        elif algorithm_dm == 'PPO':
            from training_processes.train_ppo import train_ppo as train

        elif algorithm_dm == 'DDPG':
            from training_processes.train_ddpg import train_ddpg as train

        elif algorithm_dm == 'CMA':
            from training_processes.train_cma import train_cma as train

        elif algorithm_dm == 'ODT':
            from training_processes.train_odt import train_odt as train
        
        else:
            raise RuntimeError(f'model {algorithm_dm} algorithm not found.')

        local_weights_per_agent, avg_rewards, avg_output_values, training_metrics, trajectories_per, new_buffers =\
            train(ev_info, metrics_base_path, experiment_number, chargers_copy, environment, routes, \
                  date, action_dim, global_weights, aggregate_step, ind, sub_seed, main_seed, str(device), \
                  agent_by_zone, variant, args, fixed_attributes, verbose, display_training_times, torch.float32, \
                  save_offline_data, train_model, old_buffers)

        # Save results of training
        st = time.time()
        rewards.append(avg_rewards)
        output_values.append(avg_output_values)
        metrics.append(training_metrics)
        trajectories.append(trajectories_per)
        process_buffers[ind] = new_buffers
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
        
def format_data(data):
    # Initialize a defaultdict to aggregate data by unique identifiers
    trajectories = defaultdict(lambda: {'observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'terminals_car': [], 'zone': None, 'aggregation': None, 'episode': None, 'car_idx': None})
    
    # Iterate over each data entry to aggregate the data
    for sublist in data:
        for entry in sublist:
            # Unique identifier for each car's trajectory
            identifier = (entry['zone'], entry['aggregation'], entry['episode'], entry['car_idx'])
            
            # Aggregate data for this car's trajectory
            trajectories[identifier]['observations'].extend(entry['observations'])
            trajectories[identifier]['actions'].extend(entry['actions'])
            trajectories[identifier]['rewards'].extend(entry['rewards'])
            trajectories[identifier]['terminals'].extend(entry['terminals'])
            trajectories[identifier]['terminals_car'].extend(entry['terminals_car'])  # Aggregate terminals_car
            trajectories[identifier]['zone'] = entry['zone']
            trajectories[identifier]['aggregation'] = entry['aggregation']
            trajectories[identifier]['episode'] = entry['episode']
            trajectories[identifier]['car_idx'] = entry['car_idx']
    
    # Convert the defaultdict to a list of dictionaries
    formatted_trajectories = list(trajectories.values())
    
    return formatted_trajectories
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-c','--number_processors', type=int, default=1,help='number of processors used to run MERL')
    parser.add_argument('-g','--list_gpus', nargs='*', type=int, default=[], help ='Request of enumerated gpus run MERL.')
    parser.add_argument('-e','--experiments_list', nargs='*', type=int, default=[], help ='Get the list of experiment to run.')
    parser.add_argument('-d','--data_dir', type=str, default='', help='Directory to save data to')
    parser.add_argument('-eval', type=bool, default=False, help='Evaluate the model')

    args = parser.parse_args()

    train_rl_vrp_csp(args)