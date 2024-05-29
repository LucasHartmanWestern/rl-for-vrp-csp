import torch
import torch.optim as optim
from collections import namedtuple
import random
import os
from collections import deque
from pathfinding import *
import time
from environment import simulate_matrix_env
from pathfinding import haversine
from agent import initialize, agent_learn, get_actions, soft_update, save_model
    
# Define the experience tuple
experience = namedtuple("Experience", field_names=["state", "distribution", "reward", "next_state", "done"])

def train(chargers, ev_info, routes, date, action_dim, global_weights, aggregation_num, zone_index,
    seed, main_seed, epsilon, epsilon_decay, discount_factor, learning_rate, num_episodes, batch_size,
    buffer_limit, num_of_agents, num_of_charges, layers=[64, 128, 1024, 128, 64], fixed_attributes=None,
    devices=['cpu','cpu'], verbose=False, display_training_times=False, dtype=torch.float32
):

    """
    Trains a Deep Q-Network (DQN) for Electric Vehicle (EV) routing and charging optimization.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        ev_info (dict): Information about the electric vehicles.
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
        num_of_agents (int): Number of agents (EVs) in the environment.
        num_of_charges (int): Number of charging stations.
        layers (list, optional): List of integers defining the architecture of the neural networks.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        devices (list, optional): list of two devices to run the environment and model, default both are cpu. 
                                 device[0] for environment setting, device[1] for model trainning.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.

    Returns:
        tuple: A tuple containing:
            - List of trained Q-network state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
    """

    avg_rewards = []

    step_size = 0.01  # 60km per hour / 60 minutes per hour = 1 km per minute

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(int(seed))
        dqn_rng = np.random.default_rng(seed)
    
    # Set devices for environment and agent
    device_environment = devices[0]
    device_agents = devices[1]
        
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))

    state_dimension = (num_of_charges * 3 * 2) + 3

    model_indices = ev_info['model_indices']

    q_networks = []
    target_q_networks = []
    optimizers = []

    # Assign unique NN for each agent
    for agent_ind in range(num_of_agents):
        q_network, target_q_network = initialize(state_dimension, action_dim, layers, device_agents)  # Initialize networks

        if global_weights is not None:
            q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])
            target_q_network.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])

        optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer

        # Store individual networks
        q_networks.append(q_network)
        target_q_networks.append(target_q_network)
        optimizers.append(optimizer)

    random_threshold = dqn_rng.random((num_episodes, num_of_agents))

    buffers = [deque(maxlen=buffer_limit) for _ in range(num_of_agents)]  # Initialize replay buffer with fixed size

    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None

    metrics = []

    avg_output_values = []  # List to store the average values of output neurons for each episode

    for i in range(num_episodes):  # For each episode

        paths = []
        charges_needed = []
        local_paths = []
        distributions = []
        distributions_unmodified = []
        states = []
        traffic = np.zeros(shape=(unique_chargers.shape[0], 2))

        traffic[:, 0] = unique_chargers['id']

        time_start_paths = time.time()

        # Build path for each EV
        for j in range(num_of_agents): # For each agent

            agents_chargers = chargers[j, :, 0]
            agents_unique_chargers = [charger for charger in unique_chargers if charger[0] in agents_chargers]
            agents_unique_traffic = np.array([[t[0], t[1]] for t in traffic if t[0] in agents_chargers])

            # Get distances from origin to each charging station
            org_lat, org_long, dest_lat, dest_long = routes[j]
            dists = np.array([haversine(org_lat, org_long, charge_lat, charge_long) for (id, charge_lat, charge_long) in agents_unique_chargers])
            route_dist = haversine(org_lat, org_long, dest_lat, dest_long)

            ########### GENERATE AGENT OUTPUT ###########

            t1 = time.time()

            # Traffic level and distance of each station plus total charger num, total distance, and number of EVs
            state = np.hstack((np.vstack((agents_unique_traffic[:, 1], dists)).reshape(-1), np.array([num_of_charges * 3]), np.array([route_dist]), np.array([num_of_agents])))
            states.append(state)  # Track states
            state = torch.tensor(state, dtype=dtype, device=device_agents)  # Convert state to tensor
            action_values = get_actions(state, q_networks, random_threshold, epsilon, i, j, device_agents)  # Get the action values from the agent

            t2 = time.time()

            distribution = action_values.detach().numpy()  # Convert PyTorch tensor to NumPy array
            distributions_unmodified.append(distribution.tolist()) # Track outputs before the sigmoid application
            distribution = 1 / (1 + np.exp(-distribution))  # Apply sigmoid function to the entire array
            distributions.append(distribution.tolist())  # Convert back to list and append

            t3 = time.time()

            ########### GENERATE GRAPH ###########

            # Build graph of possible paths from chargers to each other, the origin, and destination
            graph = build_graph(j, step_size, ev_info, agents_unique_chargers, org_lat, org_long, dest_lat, dest_long)
            charges_needed.append(copy.deepcopy(graph))

            t4 = time.time()

            ########### REDEFINE WEIGHTS IN GRAPH ###########

            for v in range(graph.shape[0] - 2):
                # Get multipliers from neural network
                if not fixed_attributes:
                    traffic_mult = 1 - distribution[v]
                    distance_mult = distribution[v]
                else:
                    traffic_mult = fixed_attributes[0]
                    distance_mult = fixed_attributes[1]

                # Distance * distance_mult + Traffic * traffic_mult
                graph[:, v] = graph[:, v] * distance_mult + agents_unique_traffic[v, 1] * traffic_mult

            # Set last column to zero so long as it's not infinity for every row except the last 2
            mask = (graph[:-2, -1] != np.inf)
            graph[:-2, -1][mask] = 0

            t5 = time.time()

            ########### SOLVE WEIGHTED GRAPH ###########

            path = dijkstra(graph, j)

            local_paths.append(copy.deepcopy(path))

            # Get stop ids from global list instead of only local to agent
            stop_ids = np.array([agents_unique_traffic[step, 0] for step in path])
            global_paths = np.where(np.isin(traffic[:, 0], stop_ids))[0]

            paths.append(global_paths)

            t7 = time.time()

            ########### UPDATE TRAFFIC ###########
            for step in global_paths:
                traffic[step, 1] += 1

            t8 = time.time()

            if j == 0 and display_training_times:
                print(f"Get actions - {int((t2 - t1) // 3600)}h, {int(((t2 - t1) % 3600) // 60)}m, {int((t2 - t1) % 60)}s, {int(((t2 - t1) % 1) * 1000)}ms")
                print(f"Get distributions - {int((t3 - t2) // 3600)}h, {int(((t3 - t2) % 3600) // 60)}m, {int((t3 - t2) % 60)}s, {int(((t3 - t2) % 1) * 1000)}ms")
                print(f"Build graph - {int((t4 - t3) // 3600)}h, {int(((t4 - t3) % 3600) // 60)}m, {int((t4 - t3) % 60)}s, {int(((t4 - t3) % 1) * 1000)}ms")
                print(f"Redefine weights - {int((t5 - t4) // 3600)}h, {int(((t5 - t4) % 3600) // 60)}m, {int((t5 - t4) % 60)}s, {int(((t5 - t4) % 1) * 1000)}ms")
                print(f"Solve graph and build path - {int((t7 - t5) // 3600)}h, {int(((t7 - t5) % 3600) // 60)}m, {int((t7 - t5) % 60)}s, {int(((t7 - t5) % 1) * 1000)}ms")
                print(f"Update traffic - {int((t8 - t7) // 3600)}h, {int(((t8 - t7) % 3600) // 60)}m, {int((t8 - t7) % 60)}s, {int(((t8 - t7) % 1) * 1000)}ms")


        if num_episodes == 1 and fixed_attributes is None:
            if os.path.isfile(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy'):
                paths = np.load(f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy', allow_pickle=True).tolist()

        paths_copy = copy.deepcopy(paths)

        # Calculate the average values of the output neurons for this episode
        episode_avg_output_values = np.mean(distributions_unmodified, axis=0)
        avg_output_values.append((episode_avg_output_values.tolist(), i, aggregation_num, zone_index, main_seed))

        time_end_paths = time.time() - time_start_paths

        if display_training_times:
            print(f"Get Paths - {int(time_end_paths // 3600)}h, {int((time_end_paths % 3600) // 60)}m, {int(time_end_paths % 60)}s")

        ########### GET SIMULATION RESULTS ###########

        sim_path_results, sim_traffic, sim_battery_levels, sim_distances, rewards = simulate(paths, step_size, routes, ev_info, unique_chargers, charges_needed, local_paths, device_environment)

        # Used to evaluate simulation
        metric = {
            "zone": zone_index,
            "episode": i,
            "aggregation": aggregation_num,
            "paths": sim_path_results,
            "traffic": sim_traffic,
            "batteries": sim_battery_levels,
            "distances": sim_distances,
            "rewards": rewards
        }
        metrics.append(metric)

        ########### STORE EXPERIENCES ###########

        done = True
        for d in range(len(distributions_unmodified)):
            buffers[d].append(experience(states[d], distributions_unmodified[d], rewards[d], states[(d + 1) % max(1, (len(distributions_unmodified) - 1))], done))  # Store experience

        st = time.time()

        trained = False

        for agent_ind in range(num_of_agents):

            if len(buffers[agent_ind]) >= batch_size: # Buffer is full enough

                trained = True

                mini_batch = dqn_rng.choice(np.array(buffers[agent_ind], dtype=object), batch_size, replace=False)
                experiences = map(np.stack, zip(*mini_batch))  # Format experiences
                agent_learn(experiences, discount_factor, q_networks[agent_ind], target_q_networks[agent_ind], optimizers[agent_ind], device_agents)  # Update networks

        et = time.time() - st

        if verbose and trained:
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f'Trained for {et:.3f}s', file=file)  # Print training time with 3 decimal places

            print(f'Trained for {et:.3f}s')  # Print training time with 3 decimal places

        epsilon *= epsilon_decay  # Decay epsilon
        epsilon = max(0.1, epsilon) # Minimal learning threshold

        if i % 25 == 0 and i >= buffer_limit:  # Every 25 episodes
            for agent_ind in range(num_of_agents):
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
                    print(f"Zone: {zone_index + 1} - Episode: {i} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Average Reward {round(avg_reward, 3)} - Average IR {round(avg_ir, 3)} - Epsilon: {round(epsilon, 3)}", file=file)

                print(f"Aggregation: {aggregation_num + 1} - Zone: {zone_index + 1} - Episode: {i + 1}/{num_episodes} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Average Reward {round(avg_reward, 3)} - Average IR {round(avg_ir, 3)} - Epsilon: {round(epsilon, 3)}")

    np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))
    return [q_network.cpu().state_dict() for q_network in q_networks], avg_rewards, avg_output_values, metrics

def simulate(paths, step_size, ev_routes, ev_info, unique_chargers, charge_needed, local_paths, device, dtype=torch.float64):

    """
    Simulates the EV routing and charging process to evaluate the performance of the given paths.

    Parameters:
        paths (list): List of paths for each EV.
        step_size (float): Amount to move EV each timestep.
        ev_routes (array): Array containing route information for each EV.
        ev_info (dict): Information about the electric vehicles, including usage rates.
        unique_chargers (array): Array of unique charger locations.
        charge_needed (list): List of charging requirements for each EV.
        local_paths (list): List of local paths for each EV.
        device (torch.device): Cuda davice to work with tensors.

    Returns:
        float: The simulation reward, calculated as the negative sum of the total distance traveled
               (scaled by 100) and the peak traffic encountered during the simulation.
    """

    usage_per_hour_list = ev_info['usage_per_hour']

    # Parameters to tweak
    decrease_rates = torch.Tensor(usage_per_hour_list / 60).to(device)
    increase_rate = 12500 / 60
    max_sim_steps = 500

    # Get formatted data
    tokens, destinations, capacity, stops, target_battery_level, starting_battery_level, actions, move, traffic = format_data(paths, ev_routes, ev_info, unique_chargers, charge_needed, local_paths, device, dtype)

    # Run simulation    
    path_results, traffic, battery_levels, distances = simulate_matrix_env(
        tokens, starting_battery_level, destinations, actions, move, traffic, capacity, target_battery_level, stops, step_size, increase_rate, decrease_rates, max_sim_steps, dtype=dtype)

    # Calculate reward as -(distance * 100 + peak traffic)
    simulation_reward = -(distances[-1] * 100 + np.max(traffic.numpy()))


    return path_results.numpy(), traffic.numpy(), battery_levels.numpy(), distances.numpy(), simulation_reward.numpy()

def format_data(paths, ev_routes, ev_info, unique_chargers, charge_needed, local_paths, device, dtype):

    """
    Formats the data required for simulating the EV routing and charging process.

    Parameters:
        paths (list): List of paths for each EV.
        ev_routes (array): Array containing route information for each EV.
        ev_info (dict): Information about the electric vehicles, including starting charge levels.
        unique_chargers (array): Array of unique charger locations.
        charge_needed (list): List of charging requirements for each EV.
        local_paths (list): List of local paths for each EV.
        device (torch.device): Cuda davice to work with tensors.

    Returns:
        tuple: A tuple containing:
            - tokens (torch.tensor): Tensor of origin coordinates for each EV.
            - destinations (numpy.array): Array of destination coordinates, including charging stations.
            - capacity (torch.tensor): Tensor of capacity values for each destination.
            - stops (torch.tensor): Tensor indicating the stop sequence for each EV.
            - target_battery_level (numpy.array): Array of target battery levels at each stop.
            - starting_battery_level (torch.tensor): Tensor of initial battery levels for each EV.
            - actions (numpy.array): Array of actions for each EV.
            - move (numpy.array): Array indicating movement status for each EV.
            - traffic (numpy.array): Array indicating traffic levels at each destination.
    """

    starting_charge_array = np.array(ev_info['starting_charge'], copy=True)
    starting_battery_level = torch.tensor(starting_charge_array, dtype=dtype, device=device) # 5000-7000

    tokens = torch.tensor([[o_lat, o_lon] for (o_lat, o_lon, d_lat, d_lon) in ev_routes], device=device)

    destinations = np.array([[d_lat, d_lon] for (o_lat, o_lon, d_lat, d_lon) in ev_routes])
    destinations = torch.from_numpy(destinations).to(dtype).to(device)

    stops = torch.zeros((destinations.shape[0], max(len(path) for path in paths) + 1), dtype=dtype)
    target_battery_level = torch.zeros_like(stops, device=device)

    # charging_stations = torch.zeros((len(paths),2), device=device)
    charging_stations = []
    station_ids = []

    for agent_index, path in enumerate(paths):

        prev_step = charge_needed[agent_index].shape[0] - 2

        for step_index in range(len(stops[agent_index])):

            if step_index == len(stops[agent_index]) - 1: # Go to final destination
                stops[agent_index][step_index] = agent_index + 1
                target_battery_level[agent_index, step_index] = charge_needed[agent_index][prev_step, -1]
            else:  # Go to stop
                # Check if charger already exists in list
                charger_id = unique_chargers[path[step_index]][0]

                try:
                    station_index = station_ids.index(charger_id)
                    station_index = station_index + destinations.shape[0] + 1
                except ValueError:  # Station not in list so create new station
                    station_ids.append(charger_id)
                    station_index = len(station_ids) + destinations.shape[0]
                    stop = [unique_chargers[path[step_index]][1], unique_chargers[path[step_index]][2]]  # Lat and long of charging station

                    # charging_stations[agent_index] = stop
                    charging_stations.append(stop)

                stops[agent_index][step_index] = station_index

                target_battery_level[agent_index][step_index] = charge_needed[agent_index][prev_step][local_paths[agent_index][step_index]]
                prev_step = local_paths[agent_index][step_index]

    target_battery_level = target_battery_level[:, 1:]

    # destinations = torch.vstack((destinations, charging_stations))
    charging_stations = np.array(charging_stations)
    destinations = torch.vstack((destinations, torch.from_numpy(charging_stations).to(device)))

    actions = torch.zeros((tokens.shape[0], destinations.shape[0]), device=device)
    move = torch.ones(tokens.shape[0], device=device)
    traffic = np.zeros(destinations.shape[0])

    capacity = torch.ones(len(charging_stations), dtype=dtype, device=device) * 10 # Dummy capacity of 10 cars for every station

    return tokens, destinations, capacity, stops.to(device) , target_battery_level, starting_battery_level, actions, move, traffic
