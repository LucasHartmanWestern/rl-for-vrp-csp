import torch
import torch.optim as optim
import numpy as np
import os
import time
import copy
import pickle
import h5py

from environment.evaluation import evaluate

# Replaced import from dqn_agent with reinforce_agent
from decision_makers.reinforce_agent import initialize, agent_learn, get_actions, save_model
from environment.data_loader import load_config_file
from environment._pathfinding import haversine
from .odt.odt_helpers.utils import format_data, save_to_h5, save_temp_checkpoint

def train_reinforce(ev_info, metrics_base_path, experiment_number, chargers, environment, routes, date, action_dim, global_weights, aggregation_num, zone_index,
    seed, main_seed, device, agent_by_zone, variant, args, fixed_attributes=None, verbose=False, display_training_times=False, 
          dtype=torch.float32, save_offline_data=False, train_model=True, old_buffers=None
):
    """
    Trains a policy using REINFORCE for Electric Vehicle (EV) routing and charging optimization.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        environment (dict): Class containing information about the electric vehicles.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the policy networks.
        aggregation_num (int): Aggregation step number for tracking.
        zone_index (int): Index of the current zone being processed.
        seed (int): Seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        args (argparse.Namespace): Command-line arguments.
        fixed_attributes (list, optional): List of fixed attributes for redefining weights in the graph.
        devices (list, optional): list of two devices to run the environment and model, default both are cpu.
        verbose (bool, optional): Flag to enable detailed logging.
        display_training_times (bool, optional): Flag to display training times for different operations.
        agent_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car.
        train_model (bool): True if training the model, False if evaluating.

    Returns:
        tuple: A tuple containing:
            - List of trained policy state dictionaries.
            - List of average rewards for each episode.
            - List of average output values for each episode.
            - Metrics collected during training.
    """
    # Getting Neural Network parameters
    config_fname = f'experiments/Exp_{experiment_number}/config.yaml'
    nn_c = load_config_file(config_fname)['nn_hyperparameters']
    eval_c = load_config_file(config_fname)['eval_config']
    federated_c = load_config_file(config_fname)['federated_learning_settings']

    # For policy gradient, discount factor might still be used in returns
    discount_factor = nn_c['discount_factor'] if 'discount_factor' in nn_c else 0.99
    learning_rate = nn_c['learning_rate']
    num_episodes = nn_c['num_episodes']
    layers = nn_c['layers']
    aggregation_count = federated_c['aggregation_count'] if not args.eval else federated_c['aggregation_count_eval']

    epsilon = nn_c['epsilon']
    target_episode_epsilon_frac = nn_c['target_episode_epsilon_frac'] if 'target_episode_epsilon_frac' in nn_c else 0.3

    if eval_c['evaluate_on_diff_zone'] or args.eval:
        target_episode_epsilon_frac = 0.1

    epsilon_decay =  10 ** (-1/((num_episodes * aggregation_count) * target_episode_epsilon_frac))

    epsilon = epsilon * epsilon_decay ** (num_episodes * aggregation_num)

    # For saving metrics
    eps_per_save = int(nn_c['eps_per_save']) if 'eps_per_save' in nn_c else 1

    avg_rewards = []

    # Set seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

    unique_chargers = np.unique(
        np.array(list(map(tuple, chargers.reshape(-1, 3))),
        dtype=[('id', int), ('lat', float), ('lon', float)])
    )
    state_dimension = (environment.num_chargers * 3 * 2) + 6
    model_indices = environment.info['model_indices']

    policy_networks = []
    optimizers = []
    num_cars = environment.num_cars

    # Initialize policy network(s)
    if agent_by_zone:
        num_agents = 1
        policy_net = initialize(state_dimension, action_dim, layers, device)
        if global_weights is not None:
            print("Resume from global weights")
            if eval_c['evaluate_on_diff_zone'] or args.eval:
                policy_net.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)])
            else:
                policy_net.load_state_dict(global_weights[zone_index])

        optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
        policy_networks.append(policy_net)
        optimizers.append(optimizer)
    else:
        num_agents = environment.num_cars
        for agent_ind in range(num_agents):
            policy_net = initialize(state_dimension, action_dim, layers, device)
            if global_weights is not None:
                print("Resume from global weights")
                if eval_c['evaluate_on_diff_zone'] or args.eval:
                    policy_net.load_state_dict(global_weights[(zone_index + 1) % len(global_weights)][model_indices[agent_ind]])
                else:
                    policy_net.load_state_dict(global_weights[zone_index][model_indices[agent_ind]])
            optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
            policy_networks.append(policy_net)
            optimizers.append(optimizer)

    trajectories = []
    start_time = time.time()
    best_avg = float('-inf')
    best_paths = None
    metrics = []
    avg_output_values = []  # List to store the average values of output neurons for each episode

    for i in range(num_episodes):
        if save_offline_data:
            trajectories.extend([
                {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'terminals_car': [],
                    'zone': zone_index,
                    'aggregation': aggregation_num,
                    'episode': i,      # Critical: Keep this inside the loop for correct episode tracking
                    'car_idx': car_idx
                }
                for car_idx in range(num_cars)
            ])

        distributions = []
        distributions_unmodified = []
        states = []
        rewards = []
        dones = []

        # Reset environment for this episode
        environment.reset_episode(chargers, routes, unique_chargers)
        sim_done = False
        time_start_paths = time.time()
        timestep_counter = 0
        episode_rewards = None

        # Store experiences (per-episode)
        episode_experiences = [[] for _ in range(num_agents)]

        while not sim_done:
            environment.init_routing()
            start_time_step = time.time()

            # Sample actions for each EV
            for car_idx in range(num_cars):
                if save_offline_data:
                    car_traj = next((t for t in trajectories if (
                        t['car_idx'] == car_idx and t['zone'] == zone_index and
                        t['aggregation'] == aggregation_num and t['episode'] == i
                    )), None)

                state = environment.reset_agent(car_idx, timestep_counter)
                states.append(state)
                if save_offline_data:
                    car_traj['observations'].append(state)

                # Convert state to tensor
                state_tensor = torch.tensor(state, dtype=dtype, device=device)

                # Get action distribution from policy
                if agent_by_zone:
                    action_probs = get_actions(state_tensor, policy_networks, i, car_idx, device, epsilon)
                else:
                    action_probs = get_actions(state_tensor, [policy_networks[car_idx]], i, car_idx, device, epsilon)

                distribution = action_probs

                if save_offline_data:
                    car_traj['actions'].append(distribution.detach().cpu().numpy().tolist()) #Save unmodified action
                
                distributions_unmodified.append(distribution.detach().cpu().numpy().tolist())  # Track outputs before the sigmoid application

                # Apply sigmoid function to the entire tensor
                distribution = torch.sigmoid(distribution)
                distributions.append(distribution.detach().cpu().numpy().tolist())  # Convert to list and append

                environment.generate_paths(distribution, fixed_attributes, car_idx)

            # If there's only 1 episode and no fixed attributes, we might load best paths from file
            if num_episodes == 1 and fixed_attributes is None:
                best_path_fname = f'outputs/best_paths/route_{zone_index}_seed_{main_seed}.npy'
                if os.path.isfile(best_path_fname):
                    paths = np.load(best_path_fname, allow_pickle=True).tolist()

            paths_copy = copy.deepcopy(environment.paths)

            # Track output distribution stats
            episode_avg_output_values = np.mean(distributions_unmodified, axis=0)
            avg_output_values.append((episode_avg_output_values.tolist(), i, aggregation_num, zone_index, main_seed))

            time_end_paths = time.time() - time_start_paths
            if display_training_times:
                print_time('Get Paths', time_end_paths)

            # Run simulation step
            sim_done = environment.simulate_routes(timestep_counter)
            _, sim_traffic, sim_battery_levels, sim_distances, time_step_rewards, arrived_at_final = environment.get_results()
            dones.extend(arrived_at_final.tolist())

            # Accumulate episode rewards
            if timestep_counter == 0:
                episode_rewards = np.expand_dims(time_step_rewards,axis=0)
            else:
                episode_rewards = np.vstack((episode_rewards,time_step_rewards))

            # Train the model only using the average of all timestep rewards
            if 'average_rewards_when_training' in nn_c and nn_c['average_rewards_when_training']: 
                avg_reward = time_step_rewards.sum(axis=0) / len(time_step_rewards)
                time_step_rewards_avg = [avg_reward for _ in time_step_rewards]
                rewards.extend(time_step_rewards_avg)
            # Train the model using the rewards from it's own experiences
            else:
                rewards.extend(time_step_rewards)

            # For REINFORCE, we store transitions for each agent
            for car_idx, rew in enumerate(time_step_rewards):
                episode_experiences[0 if agent_by_zone else car_idx].append((
                    states[-num_cars + car_idx],
                    distributions_unmodified[-num_cars + car_idx],
                    rew,
                    arrived_at_final[0, car_idx]
                ))

            if save_offline_data:
                arrived = environment.get_odt_info()
                for traj in trajectories:
                    if traj['episode'] == i:
                        traj['terminals'].append(sim_done)
                        car_idx = traj['car_idx']
                        traj['rewards'].append(time_step_rewards[car_idx])
                        traj['terminals_car'].append(bool(arrived[car_idx].item()))

            # Collect metrics for analysis
            metric = {
                "zone": zone_index,
                "episode": i,
                "timestep": timestep_counter,
                "aggregation": aggregation_num,
                "traffic": sim_traffic,
                "batteries": sim_battery_levels,
                "distances": sim_distances,
                "rewards": time_step_rewards,
                "best_reward": best_avg,
                "timestep_real_world_time": time.time() - start_time_step,
                "done": sim_done
            }
            metrics.append(metric)

            timestep_counter += 1
            if timestep_counter >= environment.max_steps:
                raise Exception("MAX TIME-STEPS EXCEEDED!")

        if train_model:
            st = time.time()
            for agent_ind in range(num_agents):
                # Convert stored experiences to suitable format for REINFORCE
                s_list, a_list, r_list, done_list = zip(*episode_experiences[agent_ind])
                experiences = (s_list, a_list, r_list, [None]*len(s_list), done_list)

                if agent_by_zone:
                    agent_learn(experiences, discount_factor, policy_networks[0], optimizers[0], device)
                else:
                    agent_learn(experiences, discount_factor, policy_networks[agent_ind], optimizers[agent_ind], device)

            et = time.time() - st
            if verbose:
                print(f'Spent {et // 3600}h {int((et % 3600) // 60)}m {int(et % 60)}s training')


        epsilon *= epsilon_decay  # Decay epsilon
        if train_model:
            epsilon = max(0.1, epsilon) # Minimal learning threshold

        avg_reward = episode_rewards.sum(axis=0).mean()
        avg_rewards.append((avg_reward, aggregation_num, zone_index, main_seed)) 

        if save_offline_data and (i + 1) % eps_per_save == 0:
            dataset_path = f"{metrics_base_path}/data_zone_{zone_index}.h5"
            checkpoint_dir = os.path.join(os.path.dirname(metrics_base_path), f"temp/Exp_{experiment_number}_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
        
            # Format current trajectories
            traj_format = format_data(trajectories)
        
            # Save a temp checkpoint for ODT offline data
            temp_path = os.path.join(checkpoint_dir, f"data_zone_{zone_index}_checkpoint_{(i + 1) // eps_per_save}.tmp.h5")
            with h5py.File(temp_path, 'w') as f:
                zone_grp = f.create_group(f"zone_{zone_index}")
                for i_traj, entry in enumerate(traj_format):
                    traj_grp = zone_grp.create_group(f"traj_{i_traj}")
                    for key, value in entry.items():
                        if isinstance(value, (list, np.ndarray)):
                            traj_grp.create_dataset(key, data=np.array(value))
                        else:
                            traj_grp.attrs[key] = value
        
            # Verify data
            try:
                with h5py.File(temp_path, "r") as f:
                    _ = f[f"zone_{zone_index}"]["traj_0"]["observations"][:5]
            except Exception as e:
                print(f"[ERROR] Failed to verify checkpoint (zone {zone_index}, episode {i + 1}): {e}")
                os.remove(temp_path)
                trajectories.clear()
                continue  # Skip appending and move to next episode

        
            # Append to main .h5 dataset incrementally
            with h5py.File(dataset_path, 'a') as main_f, h5py.File(temp_path, 'r') as temp_f:
                main_zone_grp = main_f.require_group(f"zone_{zone_index}")
                temp_zone_grp = temp_f[f"zone_{zone_index}"]
                existing_keys = list(main_zone_grp.keys())
                offset = len(existing_keys)
        
                for i, traj_key in enumerate(temp_zone_grp):
                    traj_data = temp_zone_grp[traj_key]
                    new_grp = main_zone_grp.create_group(f"traj_{offset + i}")
                    for key in traj_data:
                        new_grp.create_dataset(key, data=traj_data[key][:])
                    for attr_key in traj_data.attrs:
                        new_grp.attrs[attr_key] = traj_data.attrs[attr_key]
        
            os.remove(temp_path)
            trajectories.clear()

        if avg_reward > best_avg:
            best_avg = avg_reward
            best_paths = paths_copy
            if verbose:
                print(f'Zone: {zone_index + 1} - New Best: {best_avg}')

        # Some average intermediate result (avg_ir)
        avg_ir = 0
        ir_count = 0
        for distribution in distributions:
            for out in distribution:
                avg_ir += out
                ir_count += 1
        avg_ir = avg_ir / ir_count if ir_count else 0

        et = time.time() - start_time
        if verbose:
            to_print = (
                f"(Agg.: {aggregation_num + 1} - Zone: {zone_index + 1} - Episode: {i + 1}/{num_episodes})"
                f" \t et: {int(et // 3600):02d}h{int((et % 3600) // 60):02d}m{int(et % 60):02d}s -"
                f" Avg. Reward {round(avg_reward, 3):0.3f} - Time-steps: {timestep_counter}, "
                f"Epsilon: {round(epsilon, 3):0.3f}, "
                f"Avg. IR: {round(avg_ir, 3):0.3f}"
            )
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(to_print, file=file)
            print(to_print)

        # Periodically save metrics if training
        if ((i + 1) % eps_per_save == 0) or (i == num_episodes - 1):
            metrics_path = f"{metrics_base_path}/{'eval' if args.eval else 'train'}"
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)
            evaluate(ev_info, metrics, seed, date, verbose, 'save', num_episodes, f"{metrics_path}/metrics", True)
            metrics = []

    # Save best paths info
    np.save(f'outputs/best_paths/route_{zone_index}_seed_{seed}.npy', np.array(best_paths, dtype=object))

    # Return final policy states, rewards, outputs, and metrics (and old buffers since REINFORCE does not use a replay buffer)
    return [net.cpu().state_dict() for net in policy_networks], avg_rewards, avg_output_values, metrics, old_buffers

def print_time(label, elapsed_time):
    print(f"{label} - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")