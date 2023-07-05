import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from collections import namedtuple
import random


# Define a named tuple for storing experience tuples
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class QNetwork(nn.Module):
    """
    Neural Network used to approximate the Q-function.

    This class defines the architecture of the neural network
    used to approximate the Q-function in reinforcement learning.
    """

    def __init__(self, state_dim, action_dim, layers):
        super(QNetwork, self).__init__()

        # Create a ModuleList to hold the layers
        self.layers = nn.ModuleList()

        # Add input layer and hidden layers to network
        for i, layer_size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(state_dim, layer_size))  # First layer
            else:
                self.layers.append(nn.Linear(layers[i - 1], layer_size))  # Hidden layers

        # Add output layer to network
        self.layers.append(nn.Linear(layers[-1], action_dim))  # Output layer

    def forward(self, state):
        x = state
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))  # Apply ReLU activation to each layer except output

        # Output layer without activation
        return self.layers[-1](x)


def initialize_networks(state_dim, action_dim, layers):
    """
    Initialize Q-network and target Q-network with the same weights
    """

    q_network = QNetwork(state_dim, action_dim, layers)
    target_q_network = QNetwork(state_dim, action_dim, layers)
    target_q_network.load_state_dict(q_network.state_dict())

    return q_network, target_q_network


def compute_loss(experiences, epsilon, q_network, target_q_network, action_dim, discount_factor):
    """
    Compute the loss of a given set of experiences
    """

    states, actions, rewards, next_states, dones = experiences

    # Compute current Q values
    current_Q = q_network(states).gather(1, actions.unsqueeze(1))

    # Compute next action probabilities
    action_probabilities = torch.ones((next_states.shape[0], action_dim)) * epsilon / action_dim
    best_action = q_network(next_states).argmax(dim=1).unsqueeze(1)
    action_probabilities.scatter_(1, best_action, 1 - epsilon + epsilon / action_dim)

    # Compute expected Q values
    next_Q = target_q_network(next_states)
    expected_next_Q = torch.sum(next_Q * action_probabilities, dim=1).unsqueeze(1)

    # Compute target Q values
    target_Q = rewards + (discount_factor * expected_next_Q * (1 - dones))

    # Compute loss
    loss = nn.MSELoss()(current_Q, target_Q)

    return loss


def agent_learn(experiences, epsilon, q_network, target_q_network, optimizer, action_dim, discount_factor):
    """
    Implement agent learning functionality
    """

    # Convert NumPy arrays to PyTorch tensors
    states, actions, rewards, next_states, dones = experiences
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    experiences = (states, actions, rewards, next_states, dones)

    loss = compute_loss(experiences, epsilon, q_network, target_q_network, action_dim, discount_factor)  # Compute loss
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()  # Backpropagate loss
    optimizer.step()  # Update weights


def save_model(network, filename):
    """
    Save the weights of a network
    """

    torch.save(network.state_dict(), filename)


def load_model(network, filename):
    """
    Load the weights into a network
    """

    network.load_state_dict(torch.load(filename))


def train_sarsa(environment, epsilon, discount_factor, num_episodes, batch_size, buffer_limit,
                max_num_timesteps, state_dim, action_dim, load_saved=False, layers=[64, 128, 1024, 128, 64]):
    """
    Main training loop for the Sarsa agent
    """

    environment.tracking_baseline = False
    q_network, target_q_network = initialize_networks(state_dim, action_dim, layers)

    if load_saved:
        # Load saved weights if requested
        load_model(q_network, 'saved_networks/q_network.pth')
        load_model(target_q_network, 'saved_networks/target_q_network.pth')

    # Initialize the optimizer and the replay buffer
    optimizer = optim.Adam(q_network.parameters())
    buffer = []

    start_time = time.time()

    for i in range(num_episodes + 1):
        state = environment.reset()

        if i % 10 == 0:
            # Print progress every 10 episodes
            elapsed_time = time.time() - start_time
            print(f"Episode: {i} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        for j in range(max_num_timesteps):  # For each timestep
            # Convert the state to a tensor
            state = torch.tensor(state, dtype=torch.float32)

            # Choose an action using the epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(action_dim)
            else:
                action = q_network(state).argmax().item()

            # Execute the action and store the result in the replay buffer
            next_state, reward, done = environment.step(action)
            buffer.append(Experience(state, action, reward, next_state, done))

            if len(buffer) >= buffer_limit:  # If replay buffer is full enough
                mini_batch = random.sample(buffer, batch_size)  # Sample a mini-batch
                experiences = map(np.stack, zip(*mini_batch))  # Format experiences
                agent_learn(experiences, epsilon, q_network, target_q_network, optimizer, action_dim, discount_factor)  # Update networks

            if done:
                break
            state = next_state

        # Decay the epsilon value after each episode
        epsilon *= discount_factor
        # Don't let epsilon get too small
        epsilon = max(epsilon, 0.1)

        if i % 10 == 0:
            # Update the target Q-network every 10 episodes
            target_q_network.load_state_dict(q_network.state_dict())

            if not os.path.exists('saved_networks'):
                os.makedirs('saved_networks')

            # Save the Q-network and the target Q-network
            save_model(q_network, 'saved_networks/q_network.pth')
            save_model(target_q_network, 'saved_networks/target_q_network.pth')

    environment.reset()