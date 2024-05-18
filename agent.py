import torch
import torch.nn as nn
from collections import namedtuple

# Define the QNetwork architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(QNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # Add a list for batch normalization layers
        for i, layer_size in enumerate(layers):
            if i == 0:
                linear_layer = nn.Linear(state_dim, layer_size)
            else:
                linear_layer = nn.Linear(layers[i - 1], layer_size)
            
            # self.init_weights(linear_layer)  # Initialize weights and biases
            self.layers.append(linear_layer)
            self.batch_norms.append(nn.BatchNorm1d(layer_size))  # Add batch normalization layer

        self.output = nn.Linear(layers[-1], action_dim)  # Output layer
        
    def forward(self, state):
        x = state
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))  # Apply ReLU activation to each layer except output
        return self.output(x) # Output layer

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

def get_actions(state, q_networks, random_threshold, epsilon, episode_index, agent_index):

    if random_threshold[episode_index, agent_index] < epsilon:  # Epsilon-greedy action selection
        action_values = q_networks[agent_index](state)
        noise = torch.randn(action_values.size()) * epsilon  # Match the size of the action_values tensor
        action_values += noise  # Add noise for exploration
    else:
        action_values = q_networks[agent_index](state)  # Greedy action

    return action_values

def soft_update(target_network, source_network, tau=0.001):
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def save_model(network, filename):
    torch.save(network.state_dict(), filename)
