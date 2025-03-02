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
    def __init__(self, num_inputs, num_outputs, layers, std=0.5):
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
        
        # Using fixed std for more stable learning initially
        self.log_std = nn.Parameter(torch.ones(num_outputs) * np.log(std))
        
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
        
        # Output raw logits instead of sigmoid
        mu = self.actor_output(mu)
        
        # Use a smaller std for more precise actions
        std = self.log_std.exp().expand_as(mu)
        
        # Ensure std doesn't get too small to prevent numerical instability
        std = torch.clamp(std, min=1e-3, max=1.0)
        
        dist = Normal(mu, std)
        return dist, value

    def update_log_std(self, decay_rate):
        # Re-enable this method to gradually reduce exploration
        with torch.no_grad():
            self.log_std.data *= decay_rate  # Modify data in-place

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
    indices = np.random.permutation(batch_size)
    for start_idx in range(0, batch_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, batch_size)
        rand_ids = indices[start_idx:end_idx]
        yield (
            states[rand_ids],
            actions[rand_ids],
            log_probs[rand_ids],
            returns[rand_ids],
            advantages[rand_ids]
        )


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, old_log_probs, returns, advantages, epsilon, clip_param=0.2):
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy = 0
    
    # Scale entropy coefficient based on epsilon to reduce exploration as epsilon decreases
    entropy_coef = 0.02 * epsilon
    
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
            
            # Calculate log_prob directly without clamping
            new_log_prob = dist.log_prob(action).sum(axis=1)
            entropy = dist.entropy().sum(axis=1).mean()

            # Normalize advantages for more stable training
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            ratio = (new_log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            
            # Adjust clipping range based on epsilon for more conservative updates as epsilon decreases
            adaptive_clip = clip_param * (0.5 + 0.5 * epsilon)
            surr2 = torch.clamp(ratio, 1.0 - adaptive_clip, 1.0 + adaptive_clip) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (return_ - value).pow(2).mean()
            
            # Use epsilon-scaled entropy coefficient for better exploration control
            loss = critic_loss + actor_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
    
    # Return average losses for monitoring
    num_updates = ppo_epochs * (len(states) // mini_batch_size + 1)
    return {
        'actor_loss': total_actor_loss / num_updates,
        'critic_loss': total_critic_loss / num_updates,
        'entropy': total_entropy / num_updates
    }
