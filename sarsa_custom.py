# Importing necessary modules
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Defining a Q-network class that inherits from PyTorch's nn.Module
class QNetwork(nn.Module):
    # The class constructor
    def __init__(self, state_dim, action_dim, layers):
        super(QNetwork, self).__init__()  # calling the superclass's constructor
        layers_nodes = [state_dim] + layers + [action_dim]  # defining the layers for the neural network
        self.layers = nn.ModuleList()  # initializing an empty ModuleList

        # Adding layers to the module list
        for i in range(len(layers_nodes) - 1):
            self.layers.append(nn.Linear(layers_nodes[i], layers_nodes[i + 1]))  # linear transformation of the input

    # Forward propagation function
    def forward(self, x):
        # Propagating through all the layers except the last one using ReLU activation function
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)  # the output of the final layer


# Function to train a network using SARSA (State-Action-Reward-State-Action) method
def train_sarsa(environment, epsilon, discount_factor, num_episodes, epsilon_decay, max_num_timesteps, state_dim, action_dim, num_of_agents=1, load_saved=False, seed=None, layers=[64, 128, 1024, 128, 64], sim=None):

    environment.tracking_baseline = False

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    network = QNetwork(state_dim, action_dim, layers)  # initializing the Q-network
    optimizer = optim.Adam(network.parameters())  # using Adam optimizer for training
    loss_fn = nn.MSELoss()  # using Mean Square Error as the loss function

    # Loading saved network parameters if required
    if load_saved:
        network.load_state_dict(torch.load('saved_networks/q_network.pth'))

    start_time = time.time()  # marking the start time of the training

    # Training loop for each episode
    for episode in range(num_episodes):
        state = environment.reset()  # resetting the environment for the new episode
        state = torch.tensor(state, dtype=torch.float32)  # converting state to tensor
        epsilon *= epsilon_decay  # decaying epsilon
        cumulative_reward = 0  # initialize cumulative reward for the episode

        done = False
        done_counter = 0

        # Time step loop for each episode
        for t in range(max_num_timesteps * num_of_agents):

            # Update visualizer
            if sim is not None:
                sim.update_ev_position((environment.cur_lat, environment.cur_long))

            action_values = network(state)  # getting action values by forward propagation
            # creating a categorical distribution of action values
            action_distribution = torch.distributions.Categorical(logits=action_values)
            action = action_distribution.sample().item()  # sampling an action from the distribution

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.choice(action_dim)  # selecting a random action

            # Taking action in the environment
            next_state, reward, done = environment.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)  # converting next state to tensor

            # Getting next action values by forward propagation
            next_action_values = network(next_state)
            # Creating a categorical distribution of next action values
            next_action_distribution = torch.distributions.Categorical(logits=next_action_values)
            next_action = next_action_distribution.sample()  # sampling a next action from the distribution

            # Defining the target and current Q values to compute loss
            target = reward + discount_factor * torch.sum(next_action_distribution.probs * next_action_values)
            current = action_values[action]

            # Calculating loss using the loss function
            loss = loss_fn(current, target)

            # Backpropagation
            optimizer.zero_grad()  # resetting the gradients to zero
            loss.backward()  # computing gradients
            optimizer.step()  # updating parameters

            state = next_state  # updating state to next state
            cumulative_reward += reward  # adding reward to cumulative reward

            if done:  # if episode ends
                done_counter += 1

            if done_counter == num_of_agents:
                break

        # Print training information every 10 episodes
        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time  # calculating elapsed time
            print(
                f"Episode: {episode + 1} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Epsilon: {round(epsilon, 2)} - Cumulative Reward: {round(cumulative_reward, 2)}")

    # Saving the trained network's parameters
    torch.save(network.state_dict(), 'saved_networks/q_network.pth')