import torch
import torch.nn as nn
import torch.optim as optim

# Define the PolicyNetwork architecture
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            if i == 0:
                linear_layer = nn.Linear(state_dim, layer_size)
            else:
                linear_layer = nn.Linear(layers[i - 1], layer_size)
            self.layers.append(linear_layer)

        self.output = nn.Linear(layers[-1], action_dim)  # Output layer

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = torch.relu(layer(x))  # Apply ReLU activation to each layer
        return torch.softmax(self.output(x), dim=-1)  # Output layer with softmax

def initialize(state_dim, action_dim, layers, device_agents):
    policy_network = PolicyNetwork(state_dim, action_dim, layers)  # Policy network
    return policy_network.to(device_agents)

def compute_policy_loss(log_probs, rewards, gamma):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = sum([gamma ** i * rewards[i + t] for i in range(len(rewards) - t)])
        discounted_rewards.append(Gt)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    loss = -torch.sum(log_probs * discounted_rewards)
    return loss

def agent_learn(trajectories, gamma, policy_network, optimizer, device):
    log_probs = []
    rewards = []
    for trajectory in trajectories:
        for log_prob, reward in zip(trajectory['log_probs'], trajectory['rewards']):
            log_probs.append(log_prob)
            rewards.append(reward)
    log_probs = torch.stack(log_probs)
    loss = compute_policy_loss(log_probs, rewards, gamma)  # Compute policy gradient loss
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()  # Backpropagate loss
    optimizer.step()  # Update weights

def get_actions(state, policy_network, device):
    state = torch.tensor(state, dtype=torch.float32, device=device)
    action_probs = policy_network(state)
    action = torch.multinomial(action_probs, 1).item()
    log_prob = torch.log(action_probs[action])
    return action, log_prob

def save_model(network, filename):
    torch.save(network.state_dict(), filename)
