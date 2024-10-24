import os
import random
import numpy as np
import torch
import pickle
import pathlib
import wandb  # optional, only if you use Weights and Biases for logging
import csv
import time


from .evaluation.evaluate_episodes import evaluate_episode_rtg
from .models.decision_transformer import DecisionTransformer
from .training.act_trainer import ActTrainer
from .training.seq_trainer import SequenceTrainer
from collections import defaultdict

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def get_batch(agent_idx, env, agent_by_zone, trajectories, agent_buffers, state_mean, state_std, sorted_inds, 
              variant, device, scale, starting_p_sample, act_dim, batch_size=256, max_len=10, max_ep_len=10):
    if agent_by_zone == False:
        buffer = agent_buffers[agent_idx]  # Use the buffer of the corresponding agent
    else:
        buffer = trajectories
    state_dim = env.state_dim
    if variant['online_training']:
        traj_lens = np.array([len(path['observations']) for path in buffer])
        p_sample = traj_lens / sum(traj_lens)
    else:
        p_sample = starting_p_sample

    batch_inds = np.random.choice(
        np.arange(len(buffer)),
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

        # Continue with the existing operations on the data
        rewards = np.array(traj['rewards'])
        observations = np.array(traj['observations'])
        actions = np.array(traj['actions'])
        terminals = np.array(traj['terminals'])

        si = random.randint(0, rewards.shape[0] - 1)

        # get sequences from dataset
        s.append(observations[si:si + max_len].reshape(1, -1, state_dim))
        a.append(actions[si:si + max_len].reshape(1, -1, act_dim))
        r.append(rewards[si:si + max_len].reshape(1, -1, 1))
        if 'terminals' in traj:
            d.append(terminals[si:si + max_len].reshape(1, -1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
        rtg.append(discount_cumsum(rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
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

def eval_episodes(target_rew, num_eval_episodes, env, chargers, routes, act_dim, fixed_attributes, model, 
                  agent_models, agent_by_zone, max_ep_len, scale, mode, state_mean, state_std, device, 
                  use_means, eval_context):
    returns, lengths = [], []
    
    with torch.no_grad():
        ret, length = evaluate_episode_rtg(
            env, chargers, routes, act_dim, fixed_attributes, model, agent_models, agent_by_zone,
            max_ep_len=max_ep_len, scale=scale, target_return=target_rew / scale, mode=mode,
            state_mean=state_mean, state_std=state_std, device=device, use_means=use_means,
            eval_context=eval_context
        )
    ret = ret.cpu().numpy() if isinstance(ret, torch.Tensor) else ret
    length = length.cpu().numpy() if isinstance(length, torch.Tensor) else length
    returns.append(ret)
    lengths.append(length)

    return {
        f'target_{target_rew}_return_mean': np.mean(returns),
        f'target_{target_rew}_return_std': np.std(returns),
        f'target_{target_rew}_length_mean': np.mean(lengths),
        f'target_{target_rew}_length_std': np.std(lengths),
    }

def run_odt(
    devices,
    env_list,
    chargers,
    all_routes,
    act_dim,
    fixed_attributes,
    variant,
    seed,
    agent_by_zone,
    num_cars,
    exp_prefix='placeholder',
    max_ep_len=10
):
    start_time = time.time()
    if agent_by_zone:
        agent_models = None
        agent_buffers = None
        agent_optimizers = None
        agent_schedulers = None
        agent_trainers = None
    else:
        agent_buffers = {i: [] for i in range(num_cars)}  # One buffer per agent
        agent_models = {}  # Dictionary to store models for each agent
        agent_optimizers = {}
        agent_schedulers = {}
        agent_trainers = {}
        model = None

    #Target rewards set here
    if variant['online_training']:
        target_online = -45

    #Normalization factor
    scale = 1

    device = devices[0]
    dataset =  variant['dataset']
    model_type = variant['model_type']
    group_name = f'{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    off_model_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), f'')
    on_model_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), f'')

    # load dataset
    #dataset_path = f"/storage_1/epigou_storage/datasets/{dataset}.pkl"
    dataset_path = f"../Datasets/{dataset}.pkl"    
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
        returns.append(sum(path['rewards']))
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    #Clean up
    states = None  
    torch.cuda.empty_cache()  

    use_means=variant['use_action_means']
    eval_context=variant['eval_context']

    num_timesteps = sum(traj_lens)

    #Set target reward as max from dataset
    env_targets = [np.max(returns)]
    target_return = env_targets[0]

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

    for env_idx, env in enumerate(env_list):
        routes = all_routes[env_idx]
        print(f'Running in zone {env_idx}')
        state_dim = env.state_dim
        group_name = f'{dataset}_env_{env_idx}'
        exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
        log_to_wandb = variant.get('log_to_wandb', False)

        if log_to_wandb:
            # Initialize a new wandb run for each environment
            wandb.init(
                name= (f' Seed: {seed} - Zone: {env_idx}'),
                group=group_name,
                project='decision-transformer',
                config=variant
            )

        if variant['online_training']:
            # If online training, use means during eval, but (not during exploration)
            variant['use_action_means'] = True
    
    if model_type == 'dt':
        if variant['pretrained_model']:
            if agent_by_zone:
                model = torch.load(variant['pretrained_model'],map_location='cuda:1')
                model.stochastic_tanh = variant['stochastic_tanh']
                model.approximate_entropy_samples = variant['approximate_entropy_samples']
                model.to(device)
            else:
                #Make one model per car
                for car in range(num_cars):
                    agent_models[car] = torch.load(variant['pretrained_model'],map_location=device)
                    model = agent_models[car]
                    model.stochastic_tanh = variant['stochastic_tanh']
                    model.approximate_entropy_samples = variant['approximate_entropy_samples']
                    model.to(device)
        #ELSE THAN MUST BE OFFLINE TRAINING
        else:
            if agent_by_zone:
                    model = DecisionTransformer(
                    state_dim=env.state_dim,
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
            else:
                #Make one model per car
                for car in range(num_cars):
                    agent_models[car]= DecisionTransformer(
                        state_dim=env.state_dim,
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
        
        warmup_steps = variant['warmup_steps']

        if agent_by_zone:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=variant['learning_rate'],
                weight_decay=variant['weight_decay'],
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda steps: min((steps+1)/warmup_steps, 1)
            )
        else:
            for car in range(num_cars):
                agent_models[car] = agent_models[car].to(device=device)
                agent_optimizers[car] = torch.optim.AdamW(
                    agent_models[car].parameters(),
                    lr=variant['learning_rate'],
                    weight_decay=variant['weight_decay'],
                )
                agent_schedulers[car] = torch.optim.lr_scheduler.LambdaLR(
                    agent_optimizers[car],
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
            multiplier_scheduler = torch.optim.lr_scheduler.LambdaLR(
                multiplier_optimizer,
                lambda steps: min((steps+1)/warmup_steps, 1)
            )
        else:
            log_entropy_multiplier = None
            multiplier_optimizer = None
            multiplier_scheduler = None
    
        entropy_loss_fn = None
        #Loss Function creation
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

        #create trainers for each model
        if model_type == 'dt':
            if agent_by_zone:
                trainer = SequenceTrainer(
                    model=model,
                    optimizer=optimizer,
                    batch_size=batch_size,
                                            get_batch=lambda agent_idx, batch_size: get_batch(agent_idx, env, agent_by_zone, trajectories, agent_buffers, 
                                                                          state_mean, state_std, sorted_inds, variant, device, scale, 
                                                                          starting_p_sample, act_dim, batch_size, K, max_ep_len),
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    log_entropy_multiplier=log_entropy_multiplier,
                    entropy_loss_fn=entropy_loss_fn,
                    multiplier_optimizer=multiplier_optimizer,
                    multiplier_scheduler=multiplier_scheduler,
                    eval_fns=[lambda model: eval_episodes(target_return, num_eval_episodes, env, chargers, routes, act_dim, 
                                                        fixed_attributes, model, agent_models, agent_by_zone, max_ep_len, 
                                                        scale, mode, state_mean, state_std, device, use_means, eval_context)]
                )
            else:
                for car in range(num_cars):
                    agent_trainers[car] = SequenceTrainer(
                        model=agent_models[car],
                        optimizer=agent_optimizers[car],
                        batch_size=batch_size,
                        get_batch=lambda agent_idx, batch_size: get_batch(agent_idx, env, agent_by_zone, trajectories, agent_buffers, 
                                                                          state_mean, state_std, sorted_inds, variant, device, scale, 
                                                                          starting_p_sample, act_dim, batch_size, K, max_ep_len),
                        scheduler=agent_schedulers[car],
                        loss_fn=loss_fn,
                        agent_idx=car,
                        log_entropy_multiplier=log_entropy_multiplier,
                        entropy_loss_fn=entropy_loss_fn,
                        multiplier_optimizer=multiplier_optimizer,
                        multiplier_scheduler=multiplier_scheduler,
                        # Pass the single target value directly to eval_episodes
                        eval_fns=[lambda model: eval_episodes(target_return, num_eval_episodes, env, chargers, routes, act_dim, 
                                                        fixed_attributes, model, agent_models, agent_by_zone, max_ep_len, 
                                                        scale, mode, state_mean, state_std, device, use_means, eval_context)]
                    )
        if log_to_wandb:
            wandb.init(
                name= (f' Seed: {seed} - Zone: {env_idx}'),
                group=group_name,
                project='decision-transformer',
                config=variant
            )
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
                for iter in range(variant['max_iters']):  # FOR EACH EPISODE
                    # Collect new rollouts, using stochastic policy
                    ret, length, new_trajectories = evaluate_episode_rtg(
                        env,
                        chargers,
                        routes,
                        act_dim,
                        fixed_attributes,
                        model,
                        agent_models,
                        agent_by_zone,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_online / scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        use_means=False,
                        return_traj=True
                    )
        
                    num_new_trajectories = len(new_trajectories)
        
                    if agent_by_zone:       
                        if len(trajectories) + num_new_trajectories > variant['online_buffer_size']:
                            trajectories = trajectories[num_new_trajectories:]  # Remove as many old trajectories as there are new ones
                        trajectories.extend(new_trajectories)
        
                        # Perform update, eval using deterministic policy
                        outputs = trainer.train_iteration(
                            num_steps=variant['num_steps_per_iter'],
                            iter_num=iter + 1,
                            print_logs=variant['print_logs']
                        )
                        if log_to_wandb:
                            wandb.log(outputs)
                    else:
                        # If agent_by_zone is False, process car-specific trajectories

                        for idx in range(num_new_trajectories):
                            agent_buffers[idx].append(new_trajectories[idx])  
                            # Ensure the buffer maintains its maximum size
                            if len(agent_buffers[idx]) > variant['online_buffer_size']:
                                agent_buffers[idx] = agent_buffers[idx][-variant['online_buffer_size']:]  # Keep only the newest trajectories
        
                        for car_num in range(num_cars):
                            outputs = agent_trainers[car_num].train_iteration(
                                num_steps=variant['num_steps_per_iter'],
                                iter_num=iter + 1,
                                print_logs=variant['print_logs']
                            )
                            new_trajectories = None
                            ret = None
                            length = None
                            torch.cuda.empty_cache()  # Free GPU memory if applicable
                            print(f' Car num: {car_num}')
                            if log_to_wandb:
                                wandb.log(outputs)
       
            else:
                for iter in range(variant['max_iters']):
                    outputs = trainer.train_iteration(
                        num_steps=variant['num_steps_per_iter'],
                        iter_num=iter + 1,
                        print_logs=True
                    )
                    if log_to_wandb:
                        wandb.log(outputs)
        
            if log_to_wandb:
                wandb.finish()  # Close the current wandb run
        
            if variant['online_training']:
                torch.save(model, os.path.join(on_model_dir, model_type + '_' + exp_prefix + '.pt'))
            else:
                torch.save(model, os.path.join(off_model_dir, model_type + '_' + exp_prefix + '.pt'))
                
            # Stop the timer
            end_time = time.time()
            
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            
            # Display the elapsed time
            print(f"Elapsed time: {elapsed_time} seconds")


def format_data(data):
    # Initialize a defaultdict to aggregate data by unique identifiers
    trajectories = defaultdict(lambda: {'observations': [], 'actions': [], 'rewards': [], 'terminals': [], 'terminals_car': [], 'zone': None, 'aggregation': None, 'episode': None, 'car_idx': None})
    
    # Iterate over each data entry to aggregate the data
    for sublist in data:
        for entry in sublist:
            # Unique identifier for each car's trajectory
            identifier = (entry['zone'], entry['aggregation'], entry['episode'], entry['car_idx'])
            
            # Aggregate data for this car's trajectory
            trajectories[identifier]['observations'].extend(entry['observations'])
            trajectories[identifier]['actions'].extend(entry['actions'])
            trajectories[identifier]['rewards'].extend(entry['rewards'])
            trajectories[identifier]['terminals'].extend(entry['terminals'])
            trajectories[identifier]['terminals_car'].extend(entry['terminals_car'])  # Aggregate terminals_car
            trajectories[identifier]['zone'] = entry['zone']
            trajectories[identifier]['aggregation'] = entry['aggregation']
            trajectories[identifier]['episode'] = entry['episode']
            trajectories[identifier]['car_idx'] = entry['car_idx']
    
    # Convert the defaultdict to a list of dictionaries
    formatted_trajectories = list(trajectories.values())
    
    return formatted_trajectories