import torch
import torch.nn as nn
import torch.optim as optim

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(state_dim, layer_size))
            else:
                self.layers.append(nn.Linear(layers[i - 1], layer_size))
        self.output = nn.Linear(layers[-1], action_dim)
        self.output_activation = nn.Tanh()  # Assuming action space is between -1 and 1

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_activation(self.output(x))

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layers):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(state_dim + action_dim, layer_size))
            else:
                self.layers.append(nn.Linear(layers[i - 1], layer_size))
        self.output = nn.Linear(layers[-1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

def initialize(state_dim, action_dim, layers, device):
    actor = Actor(state_dim, action_dim, layers).to(device)
    critic = Critic(state_dim, action_dim, layers).to(device)
    target_actor = Actor(state_dim, action_dim, layers).to(device)
    target_critic = Critic(state_dim, action_dim, layers).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    return actor, critic, target_actor, target_critic

def ddpg_learn(experiences, gamma, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, device):
    states, actions, rewards, next_states, dones = experiences

    # Critic loss
    next_actions = target_actor(next_states)
    target_Q = rewards + (gamma * target_critic(next_states, next_actions) * (1 - dones))
    current_Q = critic(states, actions)
    critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor loss
    actor_loss = -critic(states, actor(states)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

def soft_update(target_network, source_network, tau=0.001):
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def save_model(network, filename):
    torch.save(network.state_dict(), filename)
