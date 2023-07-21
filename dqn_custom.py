import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import random
import os
import time
import copy
import heapq

from geolocation.maps_free import get_distance_and_time

# Define the QNetwork architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(QNetwork, self).__init__()

        # Create a ModuleList to hold the layers
        self.layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(state_dim, layer_size))  # First layer
            else:
                self.layers.append(nn.Linear(layers[i - 1], layer_size))  # Hidden layers

        self.layers.append(nn.Linear(layers[-1], action_dim))  # Output layer

    def forward(self, state):
        x = state
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))  # Apply ReLU activation to each layer except output
        return torch.sigmoid(self.layers[-1](x))  # Apply ReLU to the output layer to ensure output values are positive

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

    loss = nn.MSELoss()(current_Q, target_Q)  # Compute MSE loss
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
    epsilon,
    epsilon_decay,
    discount_factor,
    num_episodes,
    batch_size,
    buffer_limit,
    state_dim,
    action_dim,
    num_of_agents,
    load_saved=False,
    layers=[64, 128, 1024, 128, 64]
):
    environment.tracking_baseline = False
    q_network, target_q_network = initialize(state_dim, action_dim, layers)  # Initialize networks

    if load_saved:
        # Save the networks at the end of the episode
        load_model(q_network, 'saved_networks/q_network.pth')
        load_model(target_q_network, 'saved_networks/target_q_network.pth')

    optimizer = optim.Adam(q_network.parameters())  # Initialize optimizer
    buffer = []  # Initialize replay buffer

    start_time = time.time()

    for i in range(num_episodes):  # For each episode

        environment.reset()  # Reset environment

        paths = []
        distributions = []
        states = []
        traffic = {}

        for j in range(num_of_agents): # For each agent

            ########### GENERATE AGENT OUTPUT ###########

            state = environment.state[j]
            states.append(state)

            state = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor
            if np.random.rand() < epsilon:  # Epsilon-greedy action selection
                action_values = q_network(state) + torch.randn(action_dim) * epsilon  # add noise for exploration
            else:
                action_values = q_network(state)  # Greedy action

            distribution = action_values.tolist()
            for ir in range(len(distribution)):
                distribution[ir] = min(1, max(0, distribution[ir]))
            distributions.append(distribution)

            ########### GENERATE GRAPH ###########

            # Build graph of possible paths from chargers to each other, the origin, and destination
            verts, edges = build_graph(environment, j)
            base_edges = copy.deepcopy(edges)

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

                traffic_mult = distribution[v - 2]
                distance_mult = 1 - distribution[v - 2]

                traffic_level = 0
                charger_id = environment.charger_coords[j][verts[v] - 1][0]
                if charger_id in traffic:
                    traffic_level = traffic[charger_id]

                # Update edge weights such that the traffic and distance are given appropriate impact ratings
                for edge in edges:
                    if verts[v] in edges[edge]:
                        edges[edge][verts[v]] = distance_mult * edges[edge][verts[v]] + traffic_mult * traffic_level

            ########### SOLVE WEIGHTED GRAPH ###########
            dist, previous = dijkstra((verts, edges), 'origin')

            ########### GENERATE PATH ###########

            path = build_path(environment, base_edges, dist, previous)
            paths.append(path)

            ########### UPDATE TRAFFIC ###########
            for step in path:
                if step[0] != 'destination':
                    charger_id = environment.charger_coords[j][step[0] - 1][0]
                    if charger_id in traffic:
                        traffic[charger_id] += 1
                    else:
                        traffic[charger_id] = 1  # Initialize this charger id with a count of 1

        # For debug purposes
        # for step in paths[0]:
        #     if step[0] != 'destination':
        #         charger_id = environment.charger_coords[j][step[0] - 1][0]
        #         print(f'GO TO {charger_id} - OG {step[0] - 1} - CHARGE TO {step[1]}')
        #     else:
        #         print(f'GO TO {step[0]} - CHARGE TO {step[1]}')

        ########### GET REWARD ###########
        rewards = simulate(environment, paths)

        ########### STORE EXPERIENCES ###########
        done = True
        for d in range(len(distributions)):
            buffer.append(experience(states[d], distributions[d], rewards[d], states[(d + 1) % max(1, (len(distributions) - 1))], done))  # Store experience

        if len(buffer) >= buffer_limit:  # If replay buffer is full enough
            mini_batch = random.sample(buffer, batch_size)  # Sample a mini-batch
            experiences = map(np.stack, zip(*mini_batch))  # Format experiences
            agent_learn(experiences, discount_factor, q_network, target_q_network, optimizer)  # Update networks

        epsilon *= epsilon_decay  # Decay epsilon

        if i % 10 == 0:  # Every ten episodes
            target_q_network.load_state_dict(q_network.state_dict())  # Update target network

            # Add this before you save your model
            if not os.path.exists('saved_networks'):
                os.makedirs('saved_networks')

            # Save the networks at the end of training
            save_model(q_network, 'saved_networks/q_network.pth')
            save_model(target_q_network, 'saved_networks/target_q_network.pth')

        # Log every tenth episode
        if i % 10 == 0:
            avg_reward = 0
            for reward in rewards:
                avg_reward += reward
            avg_reward /= len(rewards)

            elapsed_time = time.time() - start_time
            print(f"Episode: {i} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Average Reward {avg_reward} - Epsilon: {epsilon}")

def build_graph(env, agent_index):
    usage_per_min = env.ev_info() / 60
    start_soc = env.base_soc
    max_soc = env.max_soc
    max_dist_from_start = start_soc / usage_per_min
    max_dist_on_full_charge = max_soc / usage_per_min

    vertices = ['origin', 'destination']
    edges = {'origin': {}, 'destination': {}}

    # Distance in minutes from destination to origin
    org_to_dest_time = get_distance_and_time((env.dest_lat[agent_index], env.dest_long[agent_index]), (env.org_lat[agent_index], env.org_long[agent_index]))[1] / 60
    if org_to_dest_time < max_dist_from_start:
        edges['origin']['destination'] = org_to_dest_time
        edges['destination']['origin'] = org_to_dest_time

    # Loop through all chargers
    for i in range(len(env.charger_coords[agent_index])):
        vertices.append(i + 1) # Track charger ID
        edges[i + 1] = {} # Add station to edges

        charger = env.charger_coords[agent_index][i]

        # Distance in minutes from charger to origin
        time_to_charger = get_distance_and_time((charger[1], charger[2]), (env.org_lat[agent_index], env.org_long[agent_index]))[1] / 60

        # If you can make it to charger from origin, log it in the graph
        if time_to_charger < max_dist_from_start:
            edges['origin'][i + 1] = time_to_charger
            edges[i + 1]['origin'] = time_to_charger

        # Distance in minutes from destination to origin
        charger_to_dest_time = get_distance_and_time((charger[1], charger[2]), (env.dest_lat[agent_index], env.dest_long[agent_index]))[1] / 60
        if charger_to_dest_time < max_dist_on_full_charge:
            edges[i + 1]['destination'] = charger_to_dest_time
            edges['destination'][i + 1] = charger_to_dest_time

        # Populate graph of individual charger
        for j in range(len(env.charger_coords[agent_index])):
            if i != j: # Ignore self reference
                other_charger = env.charger_coords[agent_index][j]

                # Distance in minutes
                time_to_other_charger = get_distance_and_time((charger[1], charger[2]), (other_charger[1], other_charger[2]))[1] / 60

                # If you can make it from one charger to another on full charge, log it
                if time_to_other_charger < max_dist_on_full_charge:
                    edges[i + 1][j + 1] = time_to_other_charger

    return vertices, edges

def dijkstra(graph, source):
    vertices, edges = graph
    dist = dict()
    previous = dict()

    for vertex in vertices:
        dist[vertex] = float('inf')
        previous[vertex] = None

    dist[source] = 0
    vertices = set(vertices)

    while vertices:
        current_vertex = min(vertices, key=lambda vertex: dist[vertex])
        vertices.remove(current_vertex)

        if dist[current_vertex] == float('inf'):
            break

        for neighbour, cost in edges[current_vertex].items():
            alternative_route = dist[current_vertex] + cost
            if alternative_route < dist[neighbour]:
                dist[neighbour] = alternative_route
                previous[neighbour] = current_vertex

    return dist, previous

def build_path(environment, edges, dist, previous):
    path = []
    usage_per_min = environment.ev_info() / 60

    # If user can make it to destination, go straight there
    if edges['origin'].get('destination') is not None:
        path.append(('destination', 0))

    # Build path based on going to chargers first
    else:
        prev = previous['destination']
        cur = 'destination'

        # Populate path to travel
        while prev != None:
            time_needed = edges[cur][prev]  # Find time needed to get to next step

            target_soc = time_needed * usage_per_min + usage_per_min  # Find SoC needed to get to next step

            path.append((cur, target_soc))  # Update path

            # Update step
            cur = copy.deepcopy(prev)
            prev = previous[prev]

    path.reverse()  # Put destination step at the end
    return path

def simulate(environment, paths):
    current_path = 0
    current_path_list = [i for i in range(len(paths))]

    simulation_reward = [0 for i in range(len(paths))]

    usage_per_min = environment.ev_info() / 60

    while len(current_path_list) > 0:

        done = False

        # Check if step in path is completed
        if len(paths[current_path_list[current_path]]) > 1:
            if environment.is_charging[current_path_list[current_path]] is True and environment.cur_soc[
                current_path_list[current_path]] > paths[current_path_list[current_path]][1][1] + usage_per_min:
                del paths[current_path_list[current_path]][0]

        if len(paths[current_path_list[current_path]]) > 0:

            if paths[current_path_list[current_path]][0][0] != 'destination':
                next_state, reward, done = environment.step(
                    paths[current_path_list[current_path]][0][0])  # Go to charger and charge until full
            else:
                next_state, reward, done = environment.step(0)

            simulation_reward[current_path_list[current_path]] += reward # Accumulate reward of every path

        if done is True:
            # For debug purposes
            # print(f'AGENT {current_path_list[current_path]} DONE - DQN')
            del current_path_list[current_path]

            if len(current_path_list) != 0:
                current_path = current_path % len(current_path_list)

        else:
            if len(current_path_list) > 0:
                current_path = (current_path + 1) % len(current_path_list)

    return simulation_reward

def save_model(network, filename):
    torch.save(network.state_dict(), filename)

def load_model(network, filename):
    network.load_state_dict(torch.load(filename))
