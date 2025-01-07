import torch
import torch.optim as optim
import numpy as np
import collections
from collections import namedtuple, deque, defaultdict
import os
import time
import copy
import pickle

from evaluation import evaluate

from agents.dqn_agent import initialize, agent_learn, get_actions, soft_update, save_model
from data_loader import load_config_file
from merl_env._pathfinding import haversine
from misc.utils import format_data

# Define the experience tuple
Experience = namedtuple("Experience", field_names=["state", "distribution", "reward", "next_state", "done"])

def train_dqn(ev_info, metrics_base_path, experiment_number, chargers, environment, routes, date, action_dim, global_weights, aggregation_num, zone_index,
    seed, main_seed, device, agent_by_zone, variant, args, fixed_attributes=None, verbose=False, display_training_times=False, 
          dtype=torch.float32, save_offline_data=False, train_model=True, old_buffers=None
):

    """
    Trains a Deep Q-Network (DQN) for Electric Vehicle (EV) routing and charging optimization.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        environment (dict): Class containing information about the electric vehicles.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the Q-networks.
        aggregation_num (int): Aggregation step number for tracking.
        zone_index (int): Index of the current zone being processed.
        seed (int): Seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        args (argparse.Namespace): Command-line arguments.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        devices (list, optional): list of two devices to run the environment and model, default both are cpu. 
                                 device[0] for environment setting, device[1] for model trainning.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.
        agent_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car
        train_model (bool): True if training the model, False if evaluating
        old_buffers (list, optional): List of old buffers to be used for experience replay.

    Returns:
        tuple: A tuple containing:
            - List of trained Q-network state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
    """
    # Getting Neural Network parameters
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    eval_c = load_config_file(config_fname)['eval_config']
    federated_c = load_config_file(config_fname)['federated_learning_settings']

    epsilon = nn_c['epsilon'] if train_model else 0
    #epsilon_decay =  nn_c['epsilon_decay']

    discount_factor = nn_c['discount_factor']
    learning_rate= nn_c['learning_rate']
    num_episodes = nn_c['num_episodes'] if train_model else 1
    batch_size   = int(nn_c['batch_size'])
    buffer_limit = int(nn_c['buffer_limit'])
    layers = nn_c['layers']
    aggregation_count = federated_c['aggregation_count']

    target_network_update_frequency = nn_c['target_network_update_frequency'] if 'target_network_update_frequency' in nn_c else 25

    eps_per_save = int(nn_c['eps_per_save'])
    
    target_episode_epsilon_frac = nn_c['target_episode_epsilon_frac'] if 'target_episode_epsilon_frac' in nn_c else 0.5

    if eval_c['evaluate_on_diff_zone'] or args.eval:
        target_episode_epsilon_frac = 0.1

    # Decay epsilon such that by the target_episode_epsilon_frac * num_episodes it is 0.1
    epsilon_decay =  10 ** (-1/((num_episodes * aggregation_count) * target_episode_epsilon_frac))

    avg_rewards = []

    # Carry over epsilon from last aggregation
    epsilon = epsilon * epsilon_decay ** (num_episodes * aggregation_num)

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        dqn_rng = np.random.default_rng(seed)
    
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))),\
                                         dtype=[('id', int), ('lat', float), ('lon', float)]))

    state_dimension = (environment.num_chargers * 3 * 2) + 6

    model_indices = environment.info['model_indices']
    
    q_networks = []
    target_q_networks = []
    optimizers = []

    num_cars = environment.num_cars
    if agent_by_zone:  # Use same NN for each zone
        # Initialize networks
        num_agents = 1
        q_network, target_q_network = initialize(state_dimension, action_dim, layers, device) 
        q_network = torch.nn.DataParallel(q_network).to(device)
        target_q_network = torch.nn.DataParallel(target_q_network).to(device)

        if global_weights is not None:
            if eval_c['evaluate_on_diff_zone'] or args.eval:
                q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)])
                target_q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)])
            else:
                q_network.load_state_dict(global_weights[zone_index])
                target_q_network.load_state_dict(global_weights[zone_index])

        optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer

        # Store individual networks
        q_networks.append(q_network)
        target_q_networks.append(target_q_network)
        optimizers.append(optimizer)

    else:  # Assign unique agent for each car
        num_agents = environment.num_cars
        for agent_ind in range(num_agents):
            # Initialize networks
            q_network, target_q_network = initialize(state_dimension, action_dim, layers, device)  
            q_network = torch.nn.DataParallel(q_network).to(device)
            target_q_network = torch.nn.DataParallel(target_q_network).to(device)

            if global_weights is not None:
                if eval_c['evaluate_on_diff_zone'] or args.eval:
                    q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)][model_indices[agent_ind]])
                    target_q_network.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)][model_indices[agent_ind]])
                else:
                    q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])
                    target_q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])

            optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer
            # Store individual networks
            q_networks.append(q_network)
            target_q_networks.append(target_q_network)
            optimizers.append(optimizer)

    random_threshold = dqn_rng.random((num_episodes, num_cars))

    buffers = [deque(maxlen=buffer_limit) for _ in range(num_cars)]  # Initialize replay buffer with fixed size

    if old_buffers is not None and len(old_buffers) > 0:
        buffers = old_buffers

    trajectories = []

    
    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None

    metrics = []

    avg_output_values = []  # List to store the average values of output neurons for each episode

    # Set model_training_batch_size from config or default to 1
    model_training_batch_size = nn_c.get('model_training_batch_size', 1)

    # Ensure device is set to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_episodes):  # For each episode

        if save_offline_data:
            for car_idx in range(num_cars):
                traj = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'terminals_car': [],
                    'zone': zone_index,
                    'aggregation': aggregation_num,
                    'episode': i,
                    'car_idx': car_idx
                }
                trajectories.append(traj)

        distributions = []
        distributions_unmodified = []
        states = []
        rewards = []
        dones = []
        # Episode includes every car reaching their destination
        environment.reset_episode(chargers, routes, unique_chargers)  

        sim_done = False

        time_start_paths = time.time()

        timestep_counter = 0
        new_rewards = []
        list_rewards= []

        while not sim_done:  # Keep going until every EV reaches its destination

            environment.init_routing()

            start_time_step = time.time()

            # Build path for each EV
            for car_idx in range(num_cars): # For each car
    
                if save_offline_data:
                    car_traj = next((traj for traj in trajectories if traj['car_idx'] == car_idx and traj['zone'] == zone_index and traj['aggregation'] == aggregation_num and traj['episode'] == i), None) #Retreive car trajectory

                ########### Starting environment rutting
                state = environment.reset_agent(car_idx, timestep_counter)
                states.append(state)  # Track states

                if save_offline_data:
                    car_traj['observations'].append(state)

                t1 = time.time()

                ####### Getting actions from agents
                state = torch.tensor(state, dtype=dtype, device=device)  # Convert state to tensor

                action_values = get_actions(state, q_networks, random_threshold, epsilon, i,\
                                            car_idx, device, agent_by_zone)  # Get the action values from the agent

                t2 = time.time()

                distribution = action_values.detach().numpy()  # Convert PyTorch tensor to NumPy array

                if save_offline_data:
                    car_traj['actions'].append(distribution.tolist()) #Save unmodified action
                
                distributions_unmodified.append(distribution.tolist())  # Track outputs before the sigmoid application

                # Apply sigmoid function to the entire array
                distribution = np.where(distribution >= 0, 1 / (1 + np.exp(-distribution)), np.exp(distribution) / (1 + np.exp(distribution)))
                distributions.append(distribution.tolist())  # Convert back to list and append

                t3 = time.time()

                environment.generate_paths(distribution, fixed_attributes, car_idx)

                t4 = time.time()

                if car_idx == 0 and display_training_times:
                    print_time("Get actions", (t2 - t1))
                    print_time("Get distributions", (t3 - t2))
                    print_time("Generate paths in environment", (t4 - t3))

            if num_episodes == 1 and fixed_attributes is None:
                if os.path.isfile(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy'):
                    paths = np.load(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy',\
                                    allow_pickle=True).tolist()

            paths_copy = copy.deepcopy(environment.paths)

            # Calculate the average values of the output neurons for this episode
            episode_avg_output_values = np.mean(distributions_unmodified, axis=0)
            avg_output_values.append((episode_avg_output_values.tolist(), i, aggregation_num,\
                                      zone_index, main_seed))

            time_end_paths = time.time() - time_start_paths

            if display_training_times:
                print_time('Get Paths', time_end_paths)

            ########### GET SIMULATION RESULTS ###########

            # Run simulation
            sim_done = environment.simulate_routes(timestep_counter)

            # Get results from environment
            _, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards, arrived_at_final = environment.get_results()
            
            dones.extend(arrived_at_final.tolist())

            if timestep_counter == 0:
                episode_rewards = np.expand_dims(time_step_rewards,axis=0)
            else:
                episode_rewards = np.vstack((episode_rewards,time_step_rewards))
            
            # Train the model only using the average of all timestep rewards
            if 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']: 
                avg_reward = time_step_rewards.sum(axis=0) / len(time_step_rewards)
                time_step_rewards_avg = [avg_reward for _ in time_step_rewards]
                rewards.extend(time_step_rewards_avg)
            # Train the model using the rewards from it's own experiences
            else:
                rewards.extend(time_step_rewards)

            time_step_time = time.time() - start_time_step

            if save_offline_data:
                arrived = environment.get_odt_info()
                for traj in trajectories:
                    car_idx = traj['car_idx']
                    if traj['episode'] == i:
                        traj['terminals'].append(sim_done)
                        traj['rewards'].append(episode_rewards[-1,car_idx])
                        traj['terminals_car'].append(bool(arrived[car_idx].item()))                

            time_step_time = time.time() - start_time_step

            # Used to evaluate simulation
            metric = {
                "zone": zone_index,
                "episode": i,
                "timestep": timestep_counter,
                "aggregation": aggregation_num,
                "traffic": sim_traffic,
                "batteries": sim_battery_levels,
                "distances": sim_distances,
                "rewards": time_step_rewards,
                "best_reward": best_avg,
                "timestep_real_world_time": time_step_time,
                "done": sim_done
            }
            metrics.append(metric)

            timestep_counter += 1  # Next timestep
            if timestep_counter >= environment.max_steps:
                raise Exception("MAX TIME-STEPS EXCEEDED!")

        ########### STORE EXPERIENCES ###########

        car_dones = [item for sublist in dones for item in sublist]

        for d in range(len(distributions_unmodified)):
            
            if d == 0:
                print(f"i: {i} - Reward: {rewards[d]}")

            buffers[d % num_cars].append(Experience(states[d], distributions_unmodified[d], rewards[d],\
                            states[(d + num_cars) if d + num_cars < len(states) else d],
                            True if car_dones[d] == 1 else False))  # Store experience

        st = time.time()

        trained = False

        # Train models in parallel batches
        for batch_start in range(0, num_cars, model_training_batch_size):
            batch_end = min(batch_start + model_training_batch_size, num_cars)
            batch_indices = range(batch_start, batch_end)

            # Collect experiences for the batch
            batch_experiences = [buffers[agent_ind] for agent_ind in batch_indices if len(buffers[agent_ind]) >= batch_size]

            # Train each model in the batch
            for agent_ind, experiences in zip(batch_indices, batch_experiences):
                mini_batch = dqn_rng.choice(np.array([Experience(exp.state.cpu().numpy(), exp.distribution, exp.reward, exp.next_state.cpu().numpy(), exp.done) if isinstance(exp.state, torch.Tensor) else exp for exp in experiences], dtype=object), batch_size, replace=False)
                experiences = map(np.stack, zip(*mini_batch))

                # Update networks
                agent_learn(experiences, discount_factor, q_networks[agent_ind], target_q_networks[agent_ind], optimizers[agent_ind], device)

        et = time.time() - st

        if verbose and trained:
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f'Trained for {et:.3f}s', file=file)  # Print training time with 3 decimal places

            print(f'Trained for {et:.3f}s')  # Print training time with 3 decimal places

        epsilon *= epsilon_decay  # Decay epsilon
        if train_model:
            epsilon = max(0.1, epsilon) # Minimal learning threshold

        base_path = f'saved_networks/Experiment {experiment_number}'

        if ((i + 1) % target_network_update_frequency == 0) and i >= batch_size:
            print(f'Updating target network at episode {i}')
            if agent_by_zone:                
                soft_update(target_q_networks[0], q_networks[0])

                # Add this before you save your model
                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                # Save the networks at the end of training
                # save_model(q_networks[0], f'{base_path}/q_network_{zone_index}.pth')
                # save_model(target_q_networks[0], f'{base_path}/target_q_network_{zone_index}.pth')
            else:
                for agent_ind in range(num_cars):
                    soft_update(target_q_networks[agent_ind], q_networks[agent_ind])

                    # Add this before you save your model
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)

                    # Save the networks at the end of training
                    # save_model(q_networks[agent_ind], f'{base_path}/q_network_{agent_ind}.pth')
                    # save_model(target_q_networks[agent_ind], f'{base_path}/target_q_network_{agent_ind}.pth')

        if ((i + 1) % eps_per_save == 0 and i > 0 and train_model) or (i == num_episodes - 2): # Save metrics data
            # Create metrics path if it does not exist
            metrics_path = f"{metrics_base_path}/{'eval' if args.eval else 'train'}"
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)

            evaluate(ev_info, metrics, seed, date, verbose, 'save', num_episodes, f"{metrics_path}/metrics", True)
            metrics = []

        #Wraping things at end of each episode
        avg_reward = episode_rewards.sum(axis=0).mean()
        # Track rewards over aggregation steps
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed)) 

        if save_offline_data and (i + 1) % eps_per_save == 0:
            # Path to the file where trajectories will be saved
            dataset_path = f"{metrics_base_path}/data_zone_{zone_index}.pkl"
            print(dataset_path)
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

            # Format the new trajectories before saving
            traj_format = format_data(trajectories)

            # Check if the file exists
            if os.path.exists(dataset_path):
                # Load existing data
                with open(dataset_path, 'rb') as f:
                    try:
                        existing_data = pickle.load(f)
                    except EOFError:
                        existing_data = []
            else:
                existing_data = []

            # Append new trajectories to existing data
            existing_data.extend(traj_format)

            # Save the combined data back to the file
            with open(dataset_path, 'wb') as f:
                pickle.dump(existing_data, f)
                print(f"Appended {len(traj_format)} trajectories to {dataset_path}. Total trajectories: {len(existing_data)}")

            
            trajectories = []


        if avg_reward > best_avg:
            best_avg = avg_reward
            best_paths = paths_copy
            if verbose:
                print(f'Zone: {zone_index + 1} - New Best: {best_avg}')

        avg_ir = 0
        ir_count = 0
        for distribution in distributions:
            for out in distribution:
                avg_ir += out
                ir_count += 1
        avg_ir /= ir_count

        et = time.time() - start_time

        # Open the file in write mode (use 'a' for append mode)
        if verbose:
            to_print = f"(Agg.: {aggregation_num + 1} - Zone: {zone_index + 1} - Episode: {i + 1}/{num_episodes})"+\
            f" \t et: {int(et // 3600):02d}h{int((et % 3600) // 60):02d}m{int(et % 60):02d}s -"+\
            f" Avg. Reward {round(avg_reward, 3):0.3f} - Time-steps: {timestep_counter}, "+\
            f"Avg. IR: {round(avg_ir, 3):0.3f} - Epsilon: {round(epsilon, 3):0.3f}"
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(to_print, file=file)

            print(to_print)

    np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))

    return [q_network.cpu().state_dict() for q_network in q_networks], avg_rewards, avg_output_values, metrics, trajectories, buffers


def print_time(label, time):
    print(f"{label} - {int(time // 3600)}h, {int((time % 3600) // 60)}m, {int(time % 60)}s")