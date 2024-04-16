import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import random
import os
from collections import deque
from pathfinding import *
import time
from MatrixEnvironment import simulate_matrix_env, visualize_simulation, visualize_stats

# Define the QNetwork architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(QNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # Add a list for batch normalization layers
        for i, layer_size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(state_dim, layer_size))
            else:
                self.layers.append(nn.Linear(layers[i - 1], layer_size))
            self.batch_norms.append(nn.BatchNorm1d(layer_size))  # Add batch normalization layer

        self.output = nn.Linear(layers[-1], action_dim)  # Output layer

    def forward(self, state):
        x = state
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = layer(x)
            x = batch_norm(x)  # Apply batch normalization
            x = torch.relu(x)
        return torch.sigmoid(self.output(x))

    def forward(self, state):
        x = state
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))  # Apply ReLU activation to each layer except output
        return self.layers[-1](x) # Output layer

# Define the experience tuple
experience = namedtuple("Experience", field_names=["state", "distribution", "reward", "next_state", "done"])

def initialize(state_dim, action_dim, layers):
    q_network = QNetwork(state_dim, action_dim, layers)  # Q-network
    target_q_network = QNetwork(state_dim, action_dim, layers)  # Target Q-network
    target_q_network.load_state_dict(q_network.state_dict())  # Initialize target Q-network with the same weights as Q-network
    return q_network, target_q_network

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, distributions, rewards, next_states, dones = experiences

    current_Q = q_network(states)
    next_Q_values = target_q_network(next_states).detach()
    max_next_Q_values = next_Q_values.max(1)[0].unsqueeze(1)
    target_Q = rewards + (gamma * max_next_Q_values * (1 - dones))

    # Expand target_Q to have the same size as current_Q
    target_Q = target_Q.expand_as(current_Q)

    # Use the Huber loss (SmoothL1Loss)
    loss = nn.SmoothL1Loss()(current_Q, target_Q)
    return loss


def agent_learn(experiences, gamma, q_network, target_q_network, optimizer):

    # Convert NumPy arrays to PyTorch tensors
    states, distributions, rewards, next_states, dones = experiences
    states = torch.tensor(states, dtype=torch.float32)
    distributions = torch.tensor(distributions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    experiences = (states, distributions, rewards, next_states, dones)

    loss = compute_loss(experiences, gamma, q_network, target_q_network)  # Compute loss
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()  # Backpropagate loss
    optimizer.step()  # Update weights

def train_dqn(
    environment,
    global_weights,
    aggregation_num,
    route_index,
    seed,
    thread_num,
    epsilon,
    epsilon_decay,
    discount_factor,
    learning_rate,
    num_episodes,
    batch_size,
    buffer_limit,
    state_dim,
    action_dim,
    num_of_agents,
    load_saved=False,
    layers=[64, 128, 1024, 128, 64],
    fixed_attributes=None
):
    avg_rewards = []

    environment.tracking_baseline = False
    q_network, target_q_network = initialize(state_dim, action_dim, layers)  # Initialize networks

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if global_weights is not None:
        q_network.load_state_dict(global_weights)
        target_q_network.load_state_dict(global_weights)
    elif load_saved:
        # Save the networks at the end of the episode
        load_model(q_network, f'saved_networks/q_network_{seed}.pth')
        load_model(target_q_network, f'saved_networks/target_q_network_{seed}.pth')

    optimizer = optim.RMSprop(q_network.parameters(), lr=learning_rate)  # Use RMSprop optimizer
    buffer = deque(maxlen=buffer_limit)  # Initialize replay buffer with fixed size

    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None

    avg_output_values = []  # List to store the average values of output neurons for each episode

    for i in range(num_episodes):  # For each episode

        environment.reset()  # Reset environment

        paths = []
        distributions = []
        distributions_unmodified = []
        states = []
        traffic = {}

        time_start_paths = time.time()

        for j in range(num_of_agents): # For each agent

            ########### GENERATE AGENT OUTPUT ###########

            t1 = time.time()

            state = environment.state[j]
            states.append(state)

            state = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor
            if np.random.rand() < epsilon:  # Epsilon-greedy action selection
                action_values = q_network(state)
                noise = torch.randn(action_values.size()) * epsilon  # Match the size of the action_values tensor
                action_values += noise  # Add noise for exploration
            else:
                action_values = q_network(state)  # Greedy action

            t2 = time.time()

            distribution = action_values.detach().numpy()  # Convert PyTorch tensor to NumPy array
            distributions_unmodified.append(distribution.tolist()) # Track outputs before the sigmoid application
            distribution = 1 / (1 + np.exp(-distribution))  # Apply sigmoid function to the entire array
            distributions.append(distribution.tolist())  # Convert back to list and append

            t3 = time.time()

            ########### GENERATE GRAPH ###########

            # Build graph of possible paths from chargers to each other, the origin, and destination
            verts, edges = build_graph(environment, j)

            base_edges = copy.deepcopy(edges)

            t4 = time.time()

            ########### REDEFINE WEIGHTS IN GRAPH ###########

            for v in range(len(verts)):
                if verts[v] == 'origin': # Ignore origin and destination
                    continue

                if verts[v] == 'destination':
                    # Update edge weights such that the traffic and distance are given appropriate impact ratings
                    for edge in edges:
                        if 'destination' in edges[edge]:
                            edges[edge][verts[v]] = 0
                    continue

                if fixed_attributes is None:
                    traffic_mult = 1 - distribution[v - 2]
                    distance_mult = distribution[v - 2]
                else:
                    traffic_mult = fixed_attributes[0]
                    distance_mult = fixed_attributes[1]

                traffic_level = 0
                charger_id = environment.charger_coords[j][verts[v] - 1][0]
                if charger_id in traffic:
                    traffic_level = traffic[charger_id]

                # Update edge weights such that the traffic and distance are given appropriate impact ratings
                for edge in edges:
                    if verts[v] in edges[edge]:
                        edges[edge][verts[v]] = distance_mult * edges[edge][verts[v]] + traffic_mult * traffic_level

            t5 = time.time()

            ########### SOLVE WEIGHTED GRAPH ###########

            dist, previous = dijkstra((verts, edges), 'origin')

            t6 = time.time()

            ########### GENERATE PATH ###########

            path = build_path(environment, base_edges, dist, previous)
            paths.append(path)

            t7 = time.time()

            ########### UPDATE TRAFFIC ###########
            for step in path:
                if step[0] != 'destination':
                    charger_id = environment.charger_coords[j][step[0] - 1][0]
                    if charger_id in traffic:
                        traffic[charger_id] += 1
                    else:
                        traffic[charger_id] = 1  # Initialize this charger id with a count of 1

            t8 = time.time()

            if j == 0 and False:
                print(f"Get actions - {int((t2 - t1) // 3600)}h, {int(((t2 - t1) % 3600) // 60)}m, {int((t2 - t1) % 60)}s, {int(((t2 - t1) % 1) * 1000)}ms")
                print(f"Get distributions - {int((t3 - t2) // 3600)}h, {int(((t3 - t2) % 3600) // 60)}m, {int((t3 - t2) % 60)}s, {int(((t3 - t2) % 1) * 1000)}ms")
                print(f"Build graph - {int((t4 - t3) // 3600)}h, {int(((t4 - t3) % 3600) // 60)}m, {int((t4 - t3) % 60)}s, {int(((t4 - t3) % 1) * 1000)}ms")
                print(f"Redefine weights - {int((t5 - t4) // 3600)}h, {int(((t5 - t4) % 3600) // 60)}m, {int((t5 - t4) % 60)}s, {int(((t5 - t4) % 1) * 1000)}ms")
                print(f"Solve graph - {int((t6 - t5) // 3600)}h, {int(((t6 - t5) % 3600) // 60)}m, {int((t6 - t5) % 60)}s, {int(((t6 - t5) % 1) * 1000)}ms")
                print(f"Build path - {int((t7 - t6) // 3600)}h, {int(((t7 - t6) % 3600) // 60)}m, {int((t7 - t6) % 60)}s, {int(((t7 - t6) % 1) * 1000)}ms")
                print(f"Update traffic - {int((t8 - t7) // 3600)}h, {int(((t8 - t7) % 3600) // 60)}m, {int((t8 - t7) % 60)}s, {int(((t8 - t7) % 1) * 1000)}ms")



        if num_episodes == 1 and fixed_attributes is None:
            if os.path.isfile(f'outputs/best_paths_{thread_num}.npy'):
                paths = np.load(f'outputs/best_paths_{thread_num}.npy', allow_pickle=True).tolist()

        paths_copy = copy.deepcopy(paths)

        # Calculate the average values of the output neurons for this episode
        episode_avg_output_values = np.mean(distributions_unmodified, axis=0)
        avg_output_values.append((episode_avg_output_values.tolist(), i, aggregation_num, route_index, seed))

        time_end_paths = time.time() - time_start_paths

        print(f"Get Paths - {int(time_end_paths // 3600)}h, {int((time_end_paths % 3600) // 60)}m, {int(time_end_paths % 60)}s")

        ########### GET REWARD ###########

        rewards = simulate(environment, paths)

        ########### STORE EXPERIENCES ###########

        done = True
        for d in range(len(distributions_unmodified)):
            buffer.append(experience(states[d], distributions_unmodified[d], rewards[d], states[(d + 1) % max(1, (len(distributions_unmodified) - 1))], done))  # Store experience

        if len(buffer) >= buffer_limit:  # If replay buffer is full enough
            st = time.time()
            mini_batch = random.sample(buffer, batch_size)  # Sample a mini-batch
            experiences = map(np.stack, zip(*mini_batch))  # Format experiences
            agent_learn(experiences, discount_factor, q_network, target_q_network, optimizer)  # Update networks
            et = time.time() - st
            print(f'Trained for {et:.3f}s')  # Print training time with 3 decimal places

        epsilon *= epsilon_decay  # Decay epsilon
        epsilon = max(0.1, epsilon) # Minimal learning threshold

        if i % 25 == 0 and i >= buffer_limit:  # Every 25 episodes
            soft_update(target_q_network, q_network)

            # Add this before you save your model
            if not os.path.exists('saved_networks'):
                os.makedirs('saved_networks')

            # Save the networks at the end of training
            save_model(q_network, f'saved_networks/q_network_{seed}.pth')
            save_model(target_q_network, f'saved_networks/target_q_network_{seed}.pth')

        # Log every ith episode
        if i % 1 == 0:
            avg_reward = 0
            for reward in rewards:
                avg_reward += reward
            avg_reward /= len(rewards)
            avg_rewards.append((avg_reward, aggregation_num, route_index, seed)) # Track rewards over aggregation steps

            if avg_reward > best_avg:
                best_avg = avg_reward
                best_paths = paths_copy
                print(f'Route Index: {route_index} - New Best: {best_avg}')

            avg_ir = 0
            ir_count = 0
            for distribution in distributions:
                for out in distribution:
                    avg_ir += out
                    ir_count += 1
            avg_ir /= ir_count

            elapsed_time = time.time() - start_time
            print(f"Route Index: {route_index} - Episode: {i} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Average Reward {round(avg_reward, 3)} - Average IR {round(avg_ir, 3)} - Epsilon: {round(epsilon, 3)}")

    np.save(f'outputs/best_paths_{thread_num}.npy', np.array(best_paths, dtype=object))
    return (q_network.state_dict(), avg_rewards, avg_output_values)

def simulate(environment, paths):

    # Parameters to tweak
    decrease_rate = 15600 / 60
    increase_rate = 12500 / 60
    step_size = 0.01 # 60km per hour / 60 minutes per hour = 1 km per minute
    max_sim_steps = 500

    t1 = time.time()

    # Get formatted data
    tokens, destinations, capacity, stops, target_battery_level, starting_battery_level, actions, move, traffic = format_data(environment, paths)

    el1 = time.time() - t1
    t2 = time.time()

    # Run simulation
    path_results, traffic, battery_levels, distances = simulate_matrix_env(
        tokens, starting_battery_level, destinations, actions, move, traffic, capacity, target_battery_level, stops, step_size, increase_rate, decrease_rate, max_sim_steps)

    el2 = time.time() - t2

    print(f"Format data - {int(el1 // 3600)}h, {int((el1 % 3600) // 60)}m, {int(el1 % 60)}s - Simulate {int(el2 // 3600)}h, {int((el2 % 3600) // 60)}m, {int(el2 % 60)}s")

    # Calculate reward as -(distance * 100 + peak traffic)
    simulation_reward = -(distances[-1] * 100 + np.max(traffic))

    return simulation_reward

def format_data(environment, paths):

    tokens = np.array([environment.org_lat, environment.org_long]).T
    destinations = np.array([environment.dest_lat, environment.dest_long]).T

    capacity = np.ones(destinations.shape[0]) * 10

    stops = np.zeros((destinations.shape[0], max(len(path) for path in paths)))
    target_battery_level = np.zeros_like(stops)

    counter = destinations.shape[0] + 1

    charging_stations = []

    starting_battery_level = [environment.cur_soc[environment.agent_list[i]] for i, _ in enumerate(paths)]

    for agent_index, path in enumerate(paths):
        for step_index, step in enumerate(path):
            if step[0] == 'destination':
                stops[agent_index][step_index] = agent_index + 1
            else:
                stop = [environment.charger_coords[environment.agent_list[agent_index]][step[0] - 1][1], environment.charger_coords[environment.agent_list[agent_index]][step[0] - 1][2]]
                charging_stations.append(stop)
                stops[agent_index][step_index] = counter
                counter += 1
            target_battery_level[agent_index][step_index] = step[1]

    destinations = np.vstack((destinations, np.array(charging_stations)))

    actions = np.zeros((tokens.shape[0], destinations.shape[0]))
    move = np.ones(tokens.shape[0])
    traffic = np.zeros(destinations.shape[0])

    return tokens, destinations, capacity, stops, target_battery_level, starting_battery_level, actions, move, traffic

def soft_update(target_network, source_network, tau=0.001):
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def save_model(network, filename):
    torch.save(network.state_dict(), filename)

def load_model(network, filename):
    network.load_state_dict(torch.load(filename))
