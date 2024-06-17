import pathlib

from train import train
from data_loader import *
from visualize import *
import random
import os
import argparse
import warnings
import time
import torch.multiprocessing as mp
from federated_learning import get_global_weights
import copy
from datetime import datetime
import numpy as np
from evaluation import evaluate
import pickle

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


mp.set_sharing_strategy('file_system')

mp.set_start_method('spawn', force=True)  # This needs to be done before you create any processes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_rl_vrp_csp(date, args):

    """
    Trains reinforcement learning models for vehicle routing and charging station placement (VRP-CSP).

    Parameters:
        date (str): The date string for logging purposes.

    Returns:
        None
    """

    ############ Initialization ############

    neural_network_config_fname = 'configs/neural_network_config.yaml'
    algorithm_config_fname = 'configs/algorithm_config.yaml'
    environment_config_fname = 'configs/environment_config.yaml'
    eval_config_fname = 'configs/evaluation_config.yaml'

    c = load_config_file(neural_network_config_fname)
    nn_c = c['nn_hyperparameters']
    c = load_config_file(algorithm_config_fname)
    algo_c = c['algorithm_settings']
    c = load_config_file(environment_config_fname)
    env_c = c['environment_settings']
    c = load_config_file(eval_config_fname)
    eval_c = c['eval_config']

    batch_size = int(nn_c['batch_size'])
    buffer_limit = int(nn_c['buffer_limit'])

    action_dim = nn_c['action_dim'] * env_c['num_of_chargers']
    
    #initializing GPUs for training
    #gpu initialization implemented to run everythin on one gpu for now. 
    #can be improved to run with multiple gpus in future
    gpus_count = len(args.list_gpus)
    if gpus_count == 0:
        devices = 'cpu' # no gpus assigned, then everything running on cpu
        print(f'Woring with CPUs for environment and model tranning')
    elif gpus_count == 1:
        devices = [f'cuda:{args.list_gpus[0]}', f'cuda:{args.list_gpus[0]}']
        print(f'Woring with one GPU, bothg environment and model tranning running with {devices[0]}')

    elif gpus_count == 2:
        devices = [f'cuda:{args.list_gpus[0]}', f'cuda:{args.list_gpus[1]}']
        print(f'Woring with two GPUs, running environment with {devices[0]} and model trainning with {devices[1]}')
    else:
        warnings.warn("**FUTURE WORK**\n Use of multiple GPUs for environment not implemented yet. Only the first GPU in given list of GPUs will be used")
        devices =  [f'cuda:{args.list_gpus[0]}', f'cuda:{args.list_gpus[0]}']

    # Run and train agents with different routes with reproducibility based on the selected seed
    for seed in env_c['seeds']:

        print(f'Running experiments with seed -> {seed}')
        # Creating and seeding a random generaton from Numpy
        rng = np.random.default_rng(seed)
        # Generating sub seeds to run on each environment
        chargers_seeds = rng.integers(low=0, high=10000, size=len(env_c['coords']))

        # Assign seed
        random.seed(seed)

        ev_info = []

        for _ in env_c['coords']:
            # Generate a random model index for each agent
            model_indices = np.array([random.randrange(3) for agent in range(env_c['num_of_agents'])], dtype=int)

            # Use the indices to select the model type and corresponding configurations
            model_type = np.array([env_c['models'][index] for index in model_indices], dtype=str)
            usage_per_hour = np.array([env_c['usage_per_hour'][index] for index in model_indices], dtype=int)
            max_charge = np.array([env_c['max_charge'][index] for index in model_indices], dtype=int)

            start_time = time.time()
            # Random charge between 0.5-x%, where x scales between 1-25% as sessions continue
            starting_charge = env_c['starting_charge'] + 2000*(rng.random(env_c['num_of_agents'])-0.5)
            elapsed_time = time.time() - start_time

            # Define a structured array
            dtypes = [('starting_charge', float),
                      ('max_charge', int),
                      ('usage_per_hour', int),
                      ('model_type', 'U50'),  # Adjust string length as needed
                      ('model_indices', int)]
            info = np.zeros(env_c['num_of_agents'], dtype=dtypes)

            # Assign values
            info['starting_charge'] = starting_charge
            info['max_charge'] = max_charge
            info['usage_per_hour'] = usage_per_hour
            info['model_type'] = model_type
            info['model_indices'] = model_indices

            ev_info.append(info)

        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get EV Info: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        all_routes = [None for route in env_c['coords']]
        for index, (city_lat, city_long) in enumerate(env_c['coords']):
            array_org_angle = rng.random(env_c['num_of_agents'])*2*np.pi # generating a list of random angles 
            all_routes[index] = get_org_dest_coords((city_lat, city_long), env_c['radius'], array_org_angle)

        elapsed_time = time.time() - start_time

        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get Routes: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        start_time = time.time()

        chargers = np.zeros(shape=[len(all_routes), env_c['num_of_agents'], env_c['num_of_chargers'] * 3, 3])

        for route_id,  route in enumerate(all_routes):
            for agent_id, (org_lat, org_long, dest_lat, dest_long) in enumerate(route):
                data = get_charger_data()
                charger_info = np.c_[data['latitude'].to_list(), data['longitude'].to_list()]
                charger_list = get_charger_list(charger_info, org_lat, org_long, dest_lat, dest_long, env_c['num_of_chargers'])
                chargers[route_id][agent_id] = charger_list

        elapsed_time = time.time() - start_time
        with open(f'logs/{date}-training_logs.txt', 'a') as file:
            print(f"Get Chargers: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s", file=file)

        print(f"Get Chargers: - {int(elapsed_time // 3600)}h, {int((elapsed_time % 3600) // 60)}m, {int(elapsed_time % 60)}s")

        user_input = ""

        while user_input != 'Done':
            if eval_c['train_model']:

                if user_input != "":
                    nn_c['num_episodes'] = int(user_input)
                    nn_c['epsilon'] = 0.1

                with open(f'logs/{date}-training_logs.txt', 'a') as file:
                    print(f"Training using Deep-Q Learning - Seed {seed}", file=file)

                print(f"Training using Deep-Q Learning - Seed {seed}")

                metrics = []  # Used to track all metrics
                rewards = []  # Array of [(avg_reward, aggregation_num, route_index, seed)]
                output_values = []  # Array of [(episode_avg_output_values, episode_number, aggregation_num, route_index, seed)]
                trajectories = []
                global_weights = None

                for aggregate_step in range(nn_c['aggregation_count']):

                    manager = mp.Manager()
                    local_weights_list = manager.list([None for _ in range(len(chargers))])
                    process_rewards = manager.list()
                    process_output_values = manager.list()
                    process_metrics = manager.list()
                    process_trajectories = manager.list()

                    # Barrier for synchronization
                    barrier = mp.Barrier(len(chargers))

                    # Creating output directory
                    folder = 'outputs/best_paths/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        
                    processes = []
                    for ind, charger_list in enumerate(chargers):
                        process = mp.Process(target=train_route, args=(
                            charger_list, ev_info[ind], all_routes[ind], date, action_dim, global_weights, aggregate_step,
                            ind, chargers_seeds[ind], seed, nn_c['epsilon'], nn_c['epsilon_decay'], nn_c['discount_factor'],
                            nn_c['learning_rate'], nn_c['num_episodes'], batch_size, buffer_limit,
                            env_c['num_of_agents'], env_c['num_of_chargers'], process_trajectories, nn_c['layers'],
                            eval_c['fixed_attributes'], local_weights_list, process_rewards, process_metrics,
                            process_output_values, barrier, devices, eval_c['verbose'], eval_c['display_training_times'], nn_c['nn_by_zone'], eval_c['save_offline_data']))
                        processes.append(process)
                        process.start()

                    print("Join Processes")

                    for process in processes:
                        process.join()

                    print("Join Weights")

                    # Aggregate the weights from all local models
                    global_weights = get_global_weights(local_weights_list, ev_info, nn_c['city_multiplier'], nn_c['zone_multiplier'], nn_c['model_multiplier'], nn_c['nn_by_zone'])

                    # Extend the main lists with the contents of the process lists
                    sorted_list = sorted([val[0] for sublist in process_rewards for val in sublist])
                    print(f'Min and Max rewards for the aggregation step: {sorted_list[0],sorted_list[-1]}')
                    rewards.extend(process_rewards)
                    output_values.extend(process_output_values)
                    metrics.extend(process_metrics)
                    trajectories.extend(process_trajectories)

                    with open(f'logs/{date}-training_logs.txt', 'a') as file:
                        print(f"\n\n############ Aggregation {aggregate_step + 1}/{nn_c['aggregation_count']} ############\n\n", file=file)

                    print(f"\n\n############ Aggregation {aggregate_step + 1}/{nn_c['aggregation_count']} ############\n\n",)

                # Plot the aggregated data
                if eval_c['save_aggregate_rewards']:
                    save_to_csv(rewards, 'outputs/rewards.csv')
                    save_to_csv(output_values, 'outputs/output_values.csv')

                    loaded_rewards = load_from_csv('outputs/rewards.csv')
                    loaded_output_values = load_from_csv('outputs/output_values.csv')

                    plot_aggregate_reward_data(loaded_rewards)
                    plot_aggregate_output_values_per_route(loaded_output_values)

            if eval_c['fixed_attributes'] != [0, 1] and eval_c['fixed_attributes'] != [1, 0] and eval_c['fixed_attributes'] != [0.5, 0.5]:
                attr_label = 'learned'
            else:
                fixed_attributes = eval_c['fixed_attributes']
                attr_label = f'{fixed_attributes[0]}_{fixed_attributes[1]}'

            # Save all metrics from training into a file
            if eval_c['save_data'] and eval_c['train_model']:
                evaluate(ev_info, metrics, seed, date, eval_c['verbose'], 'save', nn_c['num_episodes'], f"metrics/metrics_{env_c['num_of_agents']}_{nn_c['num_episodes']}_{seed}_{attr_label}")

            # Generate the plots for the various metrics
            if eval_c['generate_plots']:
                evaluate(ev_info, None, seed, date, eval_c['verbose'], 'display', nn_c['num_episodes'], f"metrics/metrics_{env_c['num_of_agents']}_{nn_c['num_episodes']}_{seed}_{attr_label}")

            if nn_c['num_episodes'] != 1 and eval_c['continue_training']:
                user_input = input("More Episodes? ")
            else:
                user_input = 'Done'

            # Save offline data to pkl file
            if eval_c['save_offline_data']:
                dataset_path = f'data/offline-data.pkl'
                with open(dataset_path, 'wb') as f:
                    pickle.dump(trajectories, f)
                    print('Offline Dataset Saved')

def train_route(chargers, ev_info, routes, date, action_dim, global_weights,
                aggregate_step, ind, sub_seed, main_seed, epsilon, epsilon_decay,
                discount_factor, learning_rate, num_episodes, batch_size,
                buffer_limit, num_of_agents, num_of_chargers, trajectories, layers, fixed_attributes,
                local_weights_list, rewards, metrics, output_values, barrier, devices,
                verbose, display_training_times, nn_by_zone, save_offline_data):

    """
    Trains a single route for the VRP-CSP problem using reinforcement learning in a multiprocessing environment.

    Parameters:
        chargers (array): Array of charger locations and their properties.
        ev_info (dict): Information about the electric vehicles.
        routes (array): Array containing route information for each EV.
        date (str): Date string for logging purposes.
        action_dim (int): Dimension of the action space.
        global_weights (array): Pre-trained weights for initializing the Q-networks.
        aggregate_step (int): Aggregation step number for tracking.
        ind (int): Index of the current process.
        sub_seed (int): Sub-seed for reproducibility of training.
        main_seed (int): Main seed for initializing the environment.
        epsilon (float): Initial exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Decay rate for the exploration rate.
        discount_factor (float): Discount factor for future rewards.
        learning_rate (float): Learning rate for the optimizer.
        num_episodes (int): Number of training episodes.
        batch_size (int): Size of the mini-batch for experience replay.
        buffer_limit (int): Maximum size of the experience replay buffer.
        num_of_agents (int): Number of agents (EVs) in the environment.
        num_of_chargers (int): Number of charging stations.
        layers (list): List of integers defining the architecture of the neural networks.
        fixed_attributes (list): List of fixed attributes for redefining weights in the graph.
        local_weights_list (list): List to store the local weights of each agent.
        rewards (list): List to store the average rewards for each episode.
        metrics (list): List to store the various metrics collected during a simulation
        output_values (list): List to store the average output values for each episode.
        barrier (multiprocessing.Barrier): Barrier for synchronizing multiprocessing tasks.
        verbose (bool): Flag to enable detailed logging.
        display_training_times (bool): Flag to display training times for different operations.
        nn_by_zone (bool): True if using one neural network for each zone, and false if using a neural network for each car


    Returns:
        None
    """

    try:
        # Create a deep copy of the environment for this thread
        chargers_copy = copy.deepcopy(chargers)

        local_weights_per_agent, avg_rewards, avg_output_values, training_metrics, trajectories_per =\
            train(chargers_copy, ev_info, routes, date, action_dim, global_weights, aggregate_step, ind, sub_seed, main_seed,
                  epsilon, epsilon_decay, discount_factor, learning_rate, num_episodes, batch_size, buffer_limit, num_of_agents,
                  num_of_chargers, layers, fixed_attributes, devices, verbose, display_training_times, torch.float32, nn_by_zone, save_offline_data)

        # Save results of training
        st = time.time()
        rewards.append(avg_rewards)
        output_values.append(avg_output_values)
        metrics.append(training_metrics)
        trajectories.append(trajectories_per)
        et = time.time() - st

        if verbose:
            with open(f'logs/{date}-training_logs.txt', 'a') as file:
                print(f'Spent {et:.3f} seconds saving results', file=file)  # Print saving time with 3 decimal places
            print(f'Spent {et:.3f} seconds saving results')  # Print saving time with 3 decimal places

        local_weights_list[ind] = local_weights_per_agent

        print(f"Thread {ind} waiting")

        barrier.wait()  # Wait for all threads to finish before proceeding

    except Exception as e:
        print(f"Error in process {ind} during aggregate step {aggregate_step}: {str(e)}")
        raise

def train_odt(
        exp_prefix,
        variant
):
    def discount_cumsum(x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)


    dataset =  variant['dataset']
    model_type = variant['model_type']
    group_name = f'{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    model_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), f'./saved_networks/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #FIX THIS CODE
    #state_dim = env.observation_space.shape[0]
    #act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/Offline-Data.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    # Sort trajectories from worst to best and cut to buffer size
    if variant['online_training']:
        trajectories = [trajectories[index] for index in sorted_inds]
        trajectories = trajectories[:variant['online_buffer_size']]
        num_trajectories = len(trajectories)

    starting_p_sample = p_sample
    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training
        if variant['online_training']:
            traj_lens = np.array([len(path['observations']) for path in trajectories])
            p_sample = traj_lens / sum(traj_lens)
        else:
            p_sample = starting_p_sample


        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            if variant['online_training']:
                traj = trajectories[batch_inds[i]]
            else:
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    if variant['online_training']:
        # If online training, use means during eval, but (not during exploration)
        variant['use_action_means'] = True

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            use_means=variant['use_action_means'],
                            eval_context=variant['eval_context']
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn


    if model_type == 'dt':
        if variant['pretrained_model']:
            model = torch.load(variant['pretrained_model'],map_location='cuda:0')
            model.stochastic_tanh = variant['stochastic_tanh']
            model.approximate_entropy_samples = variant['approximate_entropy_samples']
            model.to(device)

        else:
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len*2,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
                stochastic = variant['stochastic'],
                remove_pos_embs=variant['remove_pos_embs'],
                approximate_entropy_samples = variant['approximate_entropy_samples'],
                stochastic_tanh=variant['stochastic_tanh']
            )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if variant['online_training']:
        assert(variant['pretrained_model'] is not None), "Must specify pretrained model to perform online finetuning"
        variant['use_entropy'] = True

    if variant['online_training'] and variant['target_entropy']:
        # Setup variable and optimizer for (log of) lagrangian multiplier used for entropy constraint
        # We optimize the log of the multiplier b/c lambda >= 0
        log_entropy_multiplier = torch.zeros(1, requires_grad=True, device=device)
        multiplier_optimizer = torch.optim.AdamW(
            [log_entropy_multiplier],
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        # multiplier_optimizer = torch.optim.Adam(
        #     [log_entropy_multiplier],
        #     lr=1e-3
        #     #lr=variant['learning_rate'],
        # )
        multiplier_scheduler = torch.optim.lr_scheduler.LambdaLR(
            multiplier_optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
    else:
        log_entropy_multiplier = None
        multiplier_optimizer = None
        multiplier_scheduler = None

    entropy_loss_fn = None
    if variant['stochastic']:
        if variant['use_entropy']:
            if variant['target_entropy']:
                loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.exp(log_entropy_multiplier.detach()) * torch.mean(entropies)
                target_entropy = -act_dim
                entropy_loss_fn = lambda entropies: torch.exp(log_entropy_multiplier) * (torch.mean(entropies.detach()) - target_entropy)
            else:
                loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.mean(entropies)
        else:
            loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg,r, a_log_prob, entropies: -torch.mean(a_log_prob)
    else:
        loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg, r, a_log_prob, entropies: torch.mean((a_hat - a)**2)

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            log_entropy_multiplier=log_entropy_multiplier,
            entropy_loss_fn=entropy_loss_fn,
            multiplier_optimizer=multiplier_optimizer,
            multiplier_scheduler=multiplier_scheduler,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug
    if variant['eval_only']:
        model.eval()
        eval_fns = [eval_episodes(tar) for tar in env_targets]

        for iter_num in range(variant['max_iters']):
            logs = {}
            for eval_fn in eval_fns:
                outputs = eval_fn(model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
    else:
        if variant['online_training']:
            for iter in range(variant['max_iters']):
                # Collect new rollout, using stochastic policy
                ret, length, traj = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_online/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            use_means=False,
                            return_traj=True
                )
                # Remove oldest trajectory, add new trajectory
                trajectories = trajectories[1:]
                trajectories.append(traj)

                # Perform update, eval using deterministic policy
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                if log_to_wandb:
                    wandb.log(outputs)
        else:
            for iter in range(variant['max_iters']):
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                if log_to_wandb:
                    wandb.log(outputs)

        torch.save(model,os.path.join(model_dir, model_type + '_' + exp_prefix + '.pt'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('MERL Project'))
    parser.add_argument('-c','--number_processors', type=int, default=1,help='number of processors used to run MERL')
    parser.add_argument('-g','--list_gpus', nargs='*', type=int, default=[], help ='Request of enumerated gpus run MERL.')
    args = parser.parse_args()
    
    current_datetime = datetime.now()
    date = current_datetime.strftime('%Y-%m-%d_%H-%M')

    train_rl_vrp_csp(date, args)
