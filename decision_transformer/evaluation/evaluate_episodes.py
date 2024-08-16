import numpy as np
import torch
import os
import sys
import importlib.util

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
    
    state = env.reset_state

    # we keep all the histories on the device
    # note that the latest action and rewad will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):#

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
        env,
        chargers,
        routes,
        act_dim,
        fixed_attributes,
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

    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))
    
    env.reset_episode(chargers, routes, unique_chargers) #THIS IS RESETTING EVERY CAR
    
    model.eval()
    model.to(device=device)

    #state_mean = np.atleast_1d(state_mean)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    #state_std = np.atleast_1d(state_std)
    state_std = torch.from_numpy(state_std).to(device=device)

    # Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information
    state = env.reset_agent(0, True)

    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, env.state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    
    sim_states = []

    episode_return, episode_length = 0, 0

    sim_done = False

    ending_tokens = None
    ending_battery = None
    not_ready_to_leave = None

    timestep_counter = 0
    reward_list = []

    while not sim_done:
        #print(timestep_counter)
        if timestep_counter >= env.max_steps:
            raise Exception("MAX TIME-STEPS EXCEEDED!")
    
        if timestep_counter > 0:
            env.clear_paths()  # Clears existing paths
            env.update_starting_routes(ending_tokens)  # Sets new routes
            env.update_starting_battery(ending_battery)  # Sets starting battery to ending battery of last timestep
        
        for car in range (env.num_of_agents):# THIS WILL RUN FOR EVERY CAR
            #print(f'car: {car}')
            state = env.reset_agent(car , True)
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
    
            if not_ready_to_leave != None:  # Continuing from last timestep
                env.generate_paths(action, None, not_ready_to_leave[car], car)
            else:
                env.generate_paths(action, None, 0, car)

    
            cur_state = torch.from_numpy(state).to(device=device).reshape(1, env.state_dim)
            states = torch.cat([states, cur_state], dim=0)



        sim_done, ending_tokens, ending_battery, not_ready_to_leave, arrived_at_final = env.simulate_routes()
        sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards = env.get_results()

        time_step_rewards = torch.tensor(time_step_rewards, device=device, dtype=torch.float32)
        rewards[-1] = time_step_rewards[0]
        rewards = torch.cat([rewards, torch.tensor(time_step_rewards[1:], device=device)])

        reward = time_step_rewards.mean().item()
        reward_list.append(reward)
            
        timestep_counter += 1  # Next timestep

        #print(f'time_step_rewards: {time_step_rewards}')
        #print(f'Reward: {reward}')
        #print(f'Reward_list: {reward_list}')
        
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (timestep_counter+1)], dim=1)

        
        episode_length += 1
        episode_return = np.mean(reward_list)
        #print(f'episode_reward: {episode_return}')
        
        

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