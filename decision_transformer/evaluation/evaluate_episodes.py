import numpy as np
import torch
import os
import sys
import importlib.util

def evaluate_episode_rtg(
        env,
        chargers,
        routes,
        act_dim,
        fixed_attributes,
        model,
        agent_models,
        agent_by_zone,
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
    num_cars = env.num_cars
    
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))
    
    env.reset_episode(chargers, routes, unique_chargers) #THIS IS RESETTING EVERY CAR

    if agent_by_zone:
        model.eval()
        model.to(device=device)
    

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information
    state = env.reset_agent(0, True)

    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    
    sim_states = []

    episode_return, episode_length = 0, 0

    sim_done = False

    trajectories = []

    for car in range(num_cars):
        traj = {
            'observations': torch.zeros((0, env.state_dim), device=device, dtype=torch.float32),
            'actions': torch.zeros((0, act_dim), device=device, dtype=torch.float32),
            'rewards': torch.zeros(0, device=device, dtype=torch.float32),
            'terminals': torch.zeros(0, device=device, dtype=torch.bool),
            'terminals_car': [],
            'car_num': car
        }
        trajectories.append(traj)

    ending_tokens = None
    ending_battery = None
    not_ready_to_leave = None

    timestep_counter = 0
    cum_rewards = []
    

    while not sim_done:
        
        env.init_routing()
        
        for car in range (num_cars):# THIS WILL RUN FOR EVERY CAR

            car_traj = trajectories[car]
            
            state = env.reset_agent(car , True)

            car_traj['observations'] = torch.cat([car_traj['observations'], torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=torch.float32).detach()], dim=0)

            if agent_by_zone:
                action = model.get_action(
                    (car_traj['observations'].to(dtype=torch.float32) - state_mean) / state_std,
                    car_traj['actions'].to(dtype=torch.float32),
                    car_traj['rewards'].to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                    use_means=use_means,
                    custom_max_length=eval_context
                )
            else:
                agent_models[car].to(device=device)
                action = agent_models[car].get_action(
                    (car_traj['observations'].to(dtype=torch.float32) - state_mean) / state_std,
                    car_traj['actions'].to(dtype=torch.float32),
                    car_traj['rewards'].to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                    use_means=use_means,
                    custom_max_length=eval_context
                )
                #agent_models[car].to(device='cpu')

            car_traj['actions'] = torch.cat([car_traj['actions'], action.unsqueeze(0).detach()], dim=0)
            action_sig = torch.sigmoid(action)
            action_sig = action_sig.detach().cpu().numpy()
    
            env.generate_paths(action_sig, None, car)
            
        sim_done = env.simulate_routes()
        arrived_at_final = env.arrived_at_final
        sim_path_results, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards = env.get_results()

        for traj in trajectories:
            traj['terminals'] = torch.cat([traj['terminals'], torch.tensor([sim_done], device=device, dtype=torch.bool).detach()], dim=0)
            traj['rewards'] = torch.cat([traj['rewards'], torch.tensor([time_step_rewards[traj['car_num']]], device=device, dtype=torch.float32).detach()], dim=0)
            traj['terminals_car'].append(bool(arrived_at_final[0, traj['car_num']].item()))
        
        if timestep_counter == 0:
            episode_rewards = np.expand_dims(time_step_rewards,axis=0)
        else:
            episode_rewards = np.vstack((episode_rewards,time_step_rewards))

        avg_reward = time_step_rewards.mean().item()
            
        timestep_counter += 1  # Next timestep
        
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (avg_reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (timestep_counter+1)], dim=1)

        episode_length += 1

    total_return_per_car = np.sum(episode_rewards, axis=0)
    episode_return = np.mean(total_return_per_car)

        
    if return_traj:
        for traj in trajectories:
            traj['observations'] = traj['observations'].cpu().detach().numpy().tolist()
            traj['actions'] = traj['actions'].cpu().detach().numpy().tolist()
            traj['rewards'] = traj['rewards'].cpu().detach().numpy().tolist()
            traj['terminals'] = traj['terminals'].cpu().detach().numpy().tolist()
            traj['car_num'] = traj['car_num']

        return episode_return, episode_length, trajectories
    else:
        return episode_return, episode_length