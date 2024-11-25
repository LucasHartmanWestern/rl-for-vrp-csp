"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time

MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
    vec_env,
    eval_rtg,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    device,
    chargers,
    routes,
    num_cars,
    zone_index,
    episode_num,
    aggregation_num,
    use_mean=False,
    reward_scale=0.001,
):
    def eval_episodes_fn(model):
        target_return = [eval_rtg * reward_scale] * 1
        returns, lengths, _ , metrics = vec_evaluate_episode_rtg(
            vec_env,
            chargers,
            routes,
            state_dim,
            act_dim,
            num_cars,
            zone_index,
            episode_num,
            aggregation_num,
            model,
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
        )
        suffix = "_gm" if use_mean else ""
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(
    vec_env,
    chargers,
    routes,
    state_dim,
    act_dim,
    num_cars,
    zone_index,
    episode_num,
    aggregation_num,
    model,
    target_return: list,
    max_ep_len=10,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    use_mean=False,
):
    # Move state_mean and state_std to the device once
    state_mean = torch.tensor(state_mean, device=device) if isinstance(state_mean, np.ndarray) else state_mean
    state_std = torch.tensor(state_std, device=device) if isinstance(state_std, np.ndarray) else state_std
    
    assert len(target_return) == 1
    unique_chargers = np.unique(np.array(list(map(tuple, chargers.reshape(-1, 3))), dtype=[('id', int), ('lat', float), ('lon', float)]))
    vec_env.reset_episode(chargers, routes, unique_chargers)
    model.eval()
    model.to(device=device)

    # Set the number of environments
    num_envs = 1
    
    # Pre-allocate memory for trajectories for all cars with a fixed max length
    max_traj_len = 7  # or adjust based on expected max length
    trajectories = [{
        'observations': torch.zeros((max_traj_len, state_dim), device=device, dtype=torch.float32),
        'actions': torch.zeros((max_traj_len, act_dim), device=device, dtype=torch.float32),
        'rewards': torch.zeros(max_traj_len, device=device, dtype=torch.float32),
        'terminals': torch.zeros(max_traj_len, device=device, dtype=torch.bool),
        'car_num': car,
        'cur_len': 0  # Track the current length of each trajectory
    } for car in range(num_cars)]

    sim_done = False
    timestep_counter = 0
    best_avg = float('-inf')
    episode_rewards = []
    metrics = []

    while not sim_done:
        vec_env.init_routing()
        start_time_step = time.time()

        for car in range(num_cars):
            state = vec_env.reset_agent(car, False)
            car_traj = trajectories[car]

            # Add observation to pre-allocated tensor
            if car_traj['cur_len'] < max_traj_len:
                car_traj['observations'][car_traj['cur_len']] = torch.from_numpy(state).to(device=device, dtype=torch.float32)

            # Get model predictions
            state_pred, action_dist, reward_pred = model.get_predictions(
                (car_traj['observations'][:car_traj['cur_len'] + 1] - state_mean) / state_std,
                car_traj['actions'][:car_traj['cur_len']],
                car_traj['rewards'][:car_traj['cur_len']],
                torch.tensor(target_return, device=device).reshape(1, 1),
                torch.tensor(timestep_counter, device=device, dtype=torch.long).reshape(1, 1),
                num_envs=1,
            )

            action = action_dist.mean.reshape(1, -1, act_dim)[-1, -1, :]
            action = torch.sigmoid(action)
            car_traj['actions'][car_traj['cur_len']] = action.detach()

            # Execute action
            vec_env.generate_paths(action.cpu().numpy(), None, car)

        # Finalize simulation step
        sim_done = vec_env.simulate_routes(timestep_counter)

        # Gather results
        arrived_at_final = vec_env.arrived_at_final

        _, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards, arrived_at_final = vec_env.get_results()
        dones.extend(arrived_at_final.tolist())
        if timestep_counter == 0:
            episode_rewards = np.expand_dims(time_step_rewards,axis=0)
        else:
            episode_rewards = np.vstack((episode_rewards,time_step_rewards))
        
        # Train the model only using the average of all timestep rewards
        if 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']: 
            avg_reward = time_step_rewards.sum(axis=0).mean()
            time_step_rewards_avg = [avg_reward for _ in time_step_rewards]
            rewards.extend(time_step_rewards_avg)
        # Train the model using the rewards from it's own experiences
        else:
            rewards.extend(time_step_rewards)

        # Update rewards and terminals
        for traj in trajectories:
            if traj['cur_len'] < max_traj_len:
                traj['rewards'][traj['cur_len']] = torch.tensor(time_step_rewards[traj['car_num']], device=device, dtype=torch.float32)
                traj['terminals'][traj['cur_len']] = sim_done

            traj['cur_len'] += 1

        time_step_time = time.time() - start_time_step
            
        metric = {
            "zone": zone_index,#pass
            "episode": episode_num, #pass
            "timestep": timestep_counter,
            "aggregation": aggregation_num, #pass
            "traffic": sim_traffic,
            "batteries": sim_battery_levels,
            "distances": sim_distances,
            "rewards": time_step_rewards,
            "best_reward": best_avg,
            "timestep_real_world_time": time_step_time,
            "done": sim_done
        }
        metrics.append(metric)

        timestep_counter += 1  # Next timestep
        if timestep_counter >= environment.max_steps:
            raise Exception("MAX TIME-STEPS EXCEEDED!")
                

    #SIM COMPLETE----------------------

    
    # Calculate the average return per car
    episode_return = np.mean(np.sum(np.vstack(episode_rewards), axis=0))
    # Truncate trajectories to actual length before returning
    trajectories = [{
        'observations': traj['observations'][:traj['cur_len']].cpu(),
        'actions': traj['actions'][:traj['cur_len']].cpu(),
        'rewards': traj['rewards'][:traj['cur_len']].cpu(),
        'terminals': traj['terminals'][:traj['cur_len']].cpu(),
        'car_num': traj['car_num']
    } for traj in trajectories]

    return episode_return, timestep_counter, trajectories, metrics


