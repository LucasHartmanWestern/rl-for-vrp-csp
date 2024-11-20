import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layers, std=1.0):
        super(ActorCritic, self).__init__()
        
        # Critic network
        self.critic_layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            if i == 0:
                linear_layer = nn.Linear(num_inputs, layer_size)
            else:
                linear_layer = nn.Linear(layers[i - 1], layer_size)
            self.critic_layers.append(linear_layer)
        self.critic_output = nn.Linear(layers[-1], 1)

        # Actor network
        self.actor_layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            if i == 0:
                linear_layer = nn.Linear(num_inputs, layer_size)
            else:
                linear_layer = nn.Linear(layers[i - 1], layer_size)
            self.actor_layers.append(linear_layer)
        self.actor_output = nn.Linear(layers[-1], num_outputs)
        self.log_std = nn.Parameter(torch.ones(num_outputs) * std) 
        
        self.apply(init_weights)
        
    def forward(self, x):
        # Critic forward pass
        value = x
        for layer in self.critic_layers:
            value = torch.relu(layer(value))
        value = self.critic_output(value)

        # Actor forward pass
        mu = x
        for layer in self.actor_layers:
            mu = torch.relu(layer(mu))
        mu = self.actor_output(mu)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist, value

    # def update_log_std(self, decay_rate):
    #     self.log_std.data *= decay_rate  # Modify data in-place

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    # Detach values to prevent gradient tracking
    values = torch.cat([values, next_value.unsqueeze(0)], dim=0).detach()
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    returns = torch.stack(returns)
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield (
            states[rand_ids],        # Shape: [mini_batch_size, state_dim]
            actions[rand_ids],       # Shape: [mini_batch_size, action_dim]
            log_probs[rand_ids],     # Shape: [mini_batch_size]
            returns[rand_ids],       # Shape: [mini_batch_size]
            advantages[rand_ids]     # Shape: [mini_batch_size]
        )


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, old_log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_prob, return_, advantage in ppo_iter(
            mini_batch_size, states, actions, old_log_probs, returns, advantages):

            # Ensure tensors are on correct device
            state = state.to(next(model.parameters()).device)
            action = action.to(next(model.parameters()).device)
            old_log_prob = old_log_prob.to(next(model.parameters()).device)
            return_ = return_.to(next(model.parameters()).device)
            advantage = advantage.to(next(model.parameters()).device)

            dist, value = model(state)
            value = value.squeeze(-1)
            new_log_prob = dist.log_prob(action).sum(axis=1)
            entropy = dist.entropy().sum(axis=1).mean()

            ratio = (new_log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
