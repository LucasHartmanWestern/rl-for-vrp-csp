import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(QNetwork, self).__init__()
        layers_nodes = [state_dim] + layers + [action_dim]
        self.layers = nn.ModuleList()
        for i in range(len(layers_nodes) - 1):
            self.layers.append(nn.Linear(layers_nodes[i], layers_nodes[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


def train_sarsa(environment, epsilon, discount_factor, num_episodes, epsilon_decay,
                max_num_timesteps, state_dim, action_dim, load_saved=False, layers=[64, 128, 1024, 128, 64]):
    network = QNetwork(state_dim, action_dim, layers)
    optimizer = optim.Adam(network.parameters())
    loss_fn = nn.MSELoss()

    if load_saved:
        network.load_state_dict(torch.load('saved_networks/q_network.pth'))

    start_time = time.time()

    for episode in range(num_episodes):
        state = environment.reset()
        state = torch.tensor(state, dtype=torch.float32)
        epsilon *= epsilon_decay
        cumulative_reward = 0
        for t in range(max_num_timesteps):
            action_values = network(state)
            action_distribution = torch.distributions.Categorical(logits=action_values)
            action = action_distribution.sample().item()

            if np.random.random() < epsilon:
                action = np.random.choice(action_dim)  # Random action

            next_state, reward, done = environment.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            next_action_values = network(next_state)
            next_action_distribution = torch.distributions.Categorical(logits=next_action_values)
            next_action = next_action_distribution.sample()

            target = reward + discount_factor * torch.sum(next_action_distribution.probs * next_action_values)
            current = action_values[action]
            loss = loss_fn(current, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            cumulative_reward += reward

            if done:
                break

        if (episode + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Episode: {episode + 1} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s - Epsilon: {epsilon} - Cumulative Reward: {cumulative_reward}")

    torch.save(network.state_dict(), 'saved_networks/q_network.pth')