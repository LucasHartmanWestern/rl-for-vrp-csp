import torch
import torch.optim as optim
import numpy as np
from collections import namedtuple
import os
from collections import deque
import time
import copy

from agent import initialize, agent_learn, get_actions, soft_update, save_model, set_seed

import collections

# Define the experience tuple
experience = namedtuple("Experience", field_names=["state", "distribution", "reward", "next_state", "done"])

def train(chargers, environment, routes, date, action_dim, global_weights, aggregation_num, zone_index,
    seed, main_seed, epsilon, epsilon_decay, discount_factor, learning_rate, num_episodes, batch_size,
    buffer_limit, layers, device, fixed_attributes=None, verbose=False, display_training_times=False, dtype=torch.float32, nn_by_zone=False, save_offline_data=False
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
        epsilon (float): Initial exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Decay rate for the exploration rate.
        discount_factor (float): Discount factor for future rewards.
        learning_rate (float): Learning rate for the optimizer.
        num_episodes (int): Number of training episodes.
        batch_size (int): Size of the mini-batch for experience replay.
        buffer_limit (int): Maximum size of the experience replay buffer.
        layers (list, optional): List of integers defining the architecture of the neural networks.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        devices (list, optional): list of two devices to run the environment and model, default both are cpu. 
                                 device[0] for environment setting, device[1] for model trainning.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.
        nn_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car


    Returns:
        tuple: A tuple containing:
            - List of trained Q-network state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
    """

    avg_rewards = []

    # Carry over epsilon from last aggregation
    epsilon = epsilon * epsilon_decay ** (num_episodes * aggregation_num)

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        dqn_rng = np.random.default_rng(seed)
        set_seed(seed)
    
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))),\
                                         dtype=[('id', int), ('lat', float), ('lon', float)]))

    state_dimension = (environment.num_of_chargers * 3 * 2) + 4

    model_indices = environment.info['model_indices']
    
    q_networks = []
    target_q_networks = []
    optimizers = []

    if nn_by_zone:  # Use same NN for each zone
        # Initialize networks
        q_network, target_q_network = initialize(state_dimension, action_dim, layers, device) 

        if global_weights is not None:
            q_network.load_state_dict(global_weights[zone_index])
            target_q_network.load_state_dict(global_weights[zone_index])

        optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer

        # Store individual networks
        q_networks.append(q_network)
        target_q_networks.append(target_q_network)
        optimizers.append(optimizer)

    else:  # Assign unique NN for each agent
        for agent_ind in range(environment.num_of_agents):
            # Initialize networks
            q_network, target_q_network = initialize(state_dimension, action_dim, layers, device)  

            if global_weights is not None:
                q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])
                target_q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])

            optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer

            # Store individual networks
            q_networks.append(q_network)
            target_q_networks.append(target_q_network)
            optimizers.append(optimizer)

    random_threshold = dqn_rng.random((num_episodes, environment.num_of_agents))

    buffers = [deque(maxlen=buffer_limit) for _ in range(environment.num_of_agents)]  # Initialize replay buffer with fixed size

    trajectories = []
    
    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None

    metrics = []

    avg_output_values = []  # List to store the average values of output neurons for each episode

    for i in range(num_episodes):  # For each episode


        distributions = []
        distributions_unmodified = []
        states = []
        rewards = []
        environment.reset_episode(chargers, routes, unique_chargers)  # Episode includes every car reaching their destination

        sim_done = False

        ending_tokens = None
        ending_battery = None

        time_start_paths = time.time()

        timestep_counter = 0

        while not sim_done and timestep_counter < environment.max_steps:  # Keep going until every EV reaches its destination
            if timestep_counter > 0:
                environment.clear_paths()  # Clears existing paths
                environment.update_starting_routes(ending_tokens)  # Sets new routes
                environment.update_starting_battery(ending_battery)  # Sets starting battery to ending battery of last timestep

            # Build path for each EV
            for agent_idx in range(environment.num_of_agents): # For each agent
                ########### Starting environment rutting
                state = environment.reset_agent(agent_idx)
                states.append(state)  # Track states

                t1 = time.time()

                ####### Getting actions from agents
                state = torch.tensor(state, dtype=dtype, device=device)  # Convert state to tensor
                action_values = get_actions(state, q_networks, random_threshold, epsilon, i, agent_idx,\
                                            device, nn_by_zone)  # Get the action values from the agent

                t2 = time.time()

                distribution = action_values.detach().numpy()  # Convert PyTorch tensor to NumPy array
                distributions_unmodified.append(distribution.tolist())  # Track outputs before the sigmoid application
                distribution = 1 / (1 + np.exp(-distribution))  # Apply sigmoid function to the entire array
                distributions.append(distribution.tolist())  # Convert back to list and append

                t3 = time.time()

                environment.generate_paths(distribution, fixed_attributes)

                t4 = time.time()

                if agent_idx == 0 and display_training_times:
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
            avg_output_values.append((episode_avg_output_values.tolist(), i, aggregation_num, zone_index, main_seed))

            time_end_paths = time.time() - time_start_paths

            if display_training_times:
                print_time('Get Paths', time_end_paths)

            ########### GET SIMULATION RESULTS ###########

            # Run simulation
            sim_done, ending_tokens, ending_battery = environment.simulate_routes()

            print(f"seed {seed}, step {timestep_counter}, Sim Done: {sim_done}, tokens {ending_tokens.sum()} ")

            # Get results from environment
            sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards = environment.get_results()
            rewards.extend(time_step_rewards)

            print(f'Rewards {time_step_rewards.sum()}')

            # Used to evaluate simulation
            metric = {
                "zone": zone_index,
                "episode": i,
                "timestep": timestep_counter,
                "aggregation": aggregation_num,
                "paths": sim_path_results,
                "traffic": sim_traffic,
                "batteries": sim_battery_levels,
                "distances": sim_distances,
                "rewards": rewards,
                "done": sim_done
            }
            metrics.append(metric)

            timestep_counter += 1  # Next timestep

        ########### STORE EXPERIENCES ###########

        done = True
        for d in range(len(distributions_unmodified)):
            buffers[d % environment.num_of_agents].append(experience(states[d], distributions_unmodified[d], rewards[d], states[(d + 1) % max(1, (len(distributions_unmodified) - 1))], done))  # Store experience

            # Offline data recording for ODT
            if save_offline_data:
                traj = collections.defaultdict(list)
                traj['observations'].append(states[d])
                traj['actions'].append(distributions_unmodified[d])
                traj['rewards'].append(rewards[d])
                traj['terminals'].append(done)

                for key in traj:
                    traj[key] = np.array(traj[key])
                trajectories.append(traj)

        st = time.time()

        trained = False

        for agent_ind in range(environment.num_of_agents):

            if len(buffers[agent_ind]) >= batch_size: # Buffer is full enough

                trained = True

                mini_batch = dqn_rng.choice(np.array(buffers[agent_ind], dtype=object), batch_size, replace=False)
                experiences = map(np.stack, zip(*mini_batch))  # Format experiences

                # Update networks
                if nn_by_zone:
                    agent_learn(experiences, discount_factor, q_networks[0], target_q_networks[0], optimizers[0], device)
                else:
                    agent_learn(experiences, discount_factor, q_networks[agent_ind], target_q_networks[agent_ind], optimizers[agent_ind], device)
        
        et = time.time() - st

        if verbose and trained:
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f'Trained for {et:.3f}s', file=file)  # Print training time with 3 decimal places

            print(f'Trained for {et:.3f}s')  # Print training time with 3 decimal places

        epsilon *= epsilon_decay  # Decay epsilon
        epsilon = max(0.1, epsilon) # Minimal learning threshold

        if i % 25 == 0 and i >= buffer_limit:  # Every 25 episodes
            if nn_by_zone:
                soft_update(target_q_networks[0], q_networks[0])

                # Add this before you save your model
                if not os.path.exists('saved_networks'):
                    os.makedirs('saved_networks')

                # Save the networks at the end of training
                save_model(q_networks[0], f'saved_networks/q_network_{main_seed}_{zone_index}.pth')
                save_model(target_q_networks[0], f'saved_networks/target_q_network_{main_seed}_{zone_index}.pth')
            else:
                for agent_ind in range(environment.num_of_agents):
                    soft_update(target_q_networks[agent_ind], q_networks[agent_ind])

                    # Add this before you save your model
                    if not os.path.exists('saved_networks'):
                        os.makedirs('saved_networks')

                    # Save the networks at the end of training
                    save_model(q_networks[agent_ind], f'saved_networks/q_network_{main_seed}_{agent_ind}.pth')
                    save_model(target_q_networks[agent_ind], f'saved_networks/target_q_network_{main_seed}_{agent_ind}.pth')

        # Log every ith episode
        if i % 1 == 0:
            avg_reward = 0
            for reward in rewards:
                avg_reward += reward
            avg_reward /= len(rewards)
            avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed)) # Track rewards over aggregation steps

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

            elapsed_time = time.time() - start_time

            # Open the file in write mode (use 'a' for append mode)
            if verbose:
                with open(f'logs/{date}-training_logs.txt', 'a') as file:
                    print(f"Average Reward {round(avg_reward, 3)} - Average IR {round(avg_ir, 3)} - Epsilon: {round(epsilon, 3)}", file=file)

                print(f"Aggregation: {aggregation_num + 1} - Zone: {zone_index + 1} - Episode: {i + 1}/{num_episodes} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Average Reward {round(avg_reward, 3)} - Average IR {round(avg_ir, 3)} - Epsilon: {round(epsilon, 3)}")

    np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))

    #print(f'Trajectories {trajectories}')

    return [q_network.cpu().state_dict() for q_network in q_networks], avg_rewards, avg_output_values, metrics, trajectories


def print_time(label,time):
    print(f"{label} - {int(time // 3600)}h, {int((time % 3600) // 60)}m, {int(time % 60)}s")
    