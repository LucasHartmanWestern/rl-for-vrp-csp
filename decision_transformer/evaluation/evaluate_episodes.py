import numpy as np
import torch
import os
import sys
import importlib.util

# Define the path to the train.py file
module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'train.py')

# Load the module
spec = importlib.util.spec_from_file_location("train", module_path)
train_module = importlib.util.module_from_spec(spec)
sys.modules["train"] = train_module
spec.loader.exec_module(train_module)

# Now you can use the functions from train.py
state_generation = train_module.state_generation
get_reward = train_module.get_reward

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    
    #Resets enviornment to initial state, returns first agent observation for an episode and information, i.e. metrics, debug info
    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and rewad will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # Updates an enviornment with actions returning the next agent observation, the reward for taking that actions...
        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        chargers,
        routes,
        num_of_charges,
        ev_info,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        use_means=False,
        return_traj=False,
        eval_context=None
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information
   # state = env.reset()
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))

    
    model_indices = [entry['model_indices'] for entry in ev_info]
    state, *intermediary = state_generation(chargers, routes, unique_chargers, model_indices, num_of_charges)
    #returns state.numpy(), agents_unique_chargers, org_lat, org_long, dest_lat, dest_long, agents_unique_traffic, traffic, time_start_paths, unique_chargers, charges_needed

    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            use_means=use_means,
            custom_max_length=eval_context
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        #State = array of floats, reward = float, done = boolean
        #state, reward, done, _ = env.step(action)

        done = True
        #State is always the same for an episode
        
        dtype = [('starting_charge', '<f8'), ('max_charge', '<i8'), ('usage_per_hour', '<i8'), ('model_type', '<U50'), ('model_indices', '<i8')]
        
        # Convert the list of tuples to a structured NumPy array
        #ev_info = np.array(ev_info, dtype=dtype)
        #print(f'(evinfo: {ev_info}')
        #FIND BETTER WAY TO HANDLE EVINFO
        metrics = get_reward(action, ev_info[0], intermediary[0],intermediary[1], intermediary[2], intermediary[3], intermediary[4], intermediary[5], intermediary[6], intermediary[7], intermediary[8], intermediary[9], routes, main_seed=1234)

        for metric in metrics:
            rewards = metric["rewards"]
            reward = rewards[0] 
        print(f'reward{reward}')

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    if return_traj:
        traj = {
            'observations': states[:-1].cpu().detach().numpy(),
            'actions': actions.cpu().detach().numpy(), 
            'rewards': rewards.cpu().detach().numpy(),
            'terminals': np.zeros(episode_length, dtype=bool)
        }
        return episode_return, episode_length, traj
    else:
        return episode_return, episode_length