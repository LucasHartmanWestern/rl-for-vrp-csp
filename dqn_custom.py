import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import random
import os
import time


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
        return self.layers[-1](x)  # Output layer without activation

# Define the experience tuple
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def initialize(state_dim, action_dim, layers):
    """Initializes the Q and target-Q neural networks

    Args:
        state_dim: How many state variables are used
        action_dim: How many actions can the system choose from
        layers: Array of hidden layers and their sizes (e.g. [64, 128, 128, 64])

    Returns:
        q_network: Q network used for DQL
        target_q_network: Target Q network used for DQL
    """

    q_network = QNetwork(state_dim, action_dim, layers)  # Q-network
    target_q_network = QNetwork(state_dim, action_dim, layers)  # Target Q-network
    target_q_network.load_state_dict(q_network.state_dict())  # Initialize target Q-network with the same weights as Q-network
    return q_network, target_q_network

def compute_loss(experiences, gamma, q_network, target_q_network):
    """Compute the loss of a given set of experiences

    Args:
        experiences: Set of tuples with the structure (state, action, reward, next_state, done)
        gamma: Discount factor for DQL
        q_network: Q network used for DQL
        target_q_network: Target Q network used for DQL

    Returns:
        loss: Integer value used to train the network
    """

    states, actions, rewards, next_states, dones = experiences
    current_Q = q_network(states).gather(1, actions.unsqueeze(1))  # Q-value from Q-network
    next_Q = target_q_network(next_states).detach().max(1)[0].unsqueeze(1)  # Maximum Q-value from target Q-network
    target_Q = rewards + (gamma * next_Q * (1 - dones))  # Target Q-value
    loss = nn.MSELoss()(current_Q, target_Q)  # Compute MSE loss
    return loss

def agent_learn(experiences, gamma, q_network, target_q_network, optimizer):
    """Implement agent learning functionality

    Args:
        experiences: Set of tuples with the structure (state, action, reward, next_state, done)
        gamma: Discount factor for DQL
        q_network: Q network used for DQL
        target_q_network: Target Q network used for DQL
        optimizer: Optimizer to use for training

    Returns:
        Nothing
    """

    # Convert NumPy arrays to PyTorch tensors
    states, actions, rewards, next_states, dones = experiences
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    experiences = (states, actions, rewards, next_states, dones)

    loss = compute_loss(experiences, gamma, q_network, target_q_network)  # Compute loss
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()  # Backpropagate loss
    optimizer.step()  # Update weights

def train_dqn(
        environment,
        epsilon,
        discount_factor,
        num_episodes,
        batch_size,
        buffer_limit,
        max_num_timesteps,
        state_dim,
        action_dim,
        load_saved=False,
        layers=[64, 128, 1024, 128, 64]
):
    """Main training loop

    Args:
        environment: Simulation environment which is used to get the reward and simulate actions
        epsilon: Measure for how often model will take a random action instead of the optimal one
        discount_factor: Factor to decrease the value of rewards with increasing timesteps
        num_episodes: How many times to run the simulation
        batch_size: Size of mini batches
        buffer_limit: Size of replay buffer
        max_num_timesteps: Max amount of steps to take in simulation before ending
        state_dim: How many state variables are used
        action_dim: How many actions can the system choose from
        load_saved: Reload last training model at start of training process
        layers: Array of hidden layers and their sizes (e.g. [64, 128, 128, 64])

    Returns:
        Nothing
    """
    environment.tracking_baseline = False
    q_network, target_q_network = initialize(state_dim, action_dim, layers)  # Initialize networks

    if load_saved:
        # Save the networks at the end of the episode
        load_model(q_network, 'saved_networks/q_network.pth')
        load_model(target_q_network, 'saved_networks/target_q_network.pth')

    optimizer = optim.Adam(q_network.parameters())  # Initialize optimizer
    buffer = []  # Initialize replay buffer

    start_time = time.time()

    for i in range(num_episodes + 1):  # For each episode
        state = environment.reset()  # Reset environment

        # Log every tenth episode
        if i % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode: {i} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        for j in range(max_num_timesteps):  # For each timestep
            state = torch.tensor(state, dtype=torch.float32)  # Convert state to tensor
            if np.random.rand() < epsilon:  # Epsilon-greedy action selection
                action = np.random.choice(action_dim)  # Random action
            else:
                action = q_network(state).argmax().item()  # Greedy action

            next_state, reward, done = environment.step(action)  # Execute action
            buffer.append(experience(state, action, reward, next_state, done))  # Store experience

            if len(buffer) >= buffer_limit:  # If replay buffer is full enough
                mini_batch = random.sample(buffer, batch_size)  # Sample a mini-batch
                experiences = map(np.stack, zip(*mini_batch))  # Format experiences
                agent_learn(experiences, discount_factor, q_network, target_q_network, optimizer)  # Update networks

            if done:  # If episode is done
                break
            state = next_state  # Update state

        epsilon *= discount_factor  # Decay epsilon

        if i % 10 == 0:  # Every ten episodes
            target_q_network.load_state_dict(q_network.state_dict())  # Update target network

            # Add this before you save your model
            if not os.path.exists('saved_networks'):
                os.makedirs('saved_networks')

            # Save the networks at the end of training
            save_model(q_network, 'saved_networks/q_network.pth')
            save_model(target_q_network, 'saved_networks/target_q_network.pth')

    environment.reset()  # Reset environment

def save_model(network, filename):
    torch.save(network.state_dict(), filename)

def load_model(network, filename):
    network.load_state_dict(torch.load(filename))
