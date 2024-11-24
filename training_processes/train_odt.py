"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import sys
sys.path.append('/home/epigou/rl-for-vrp-csp/training_processes/misc')
from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import torch
import numpy as np
import re
import os
import json

from evaluation import evaluate
from training_processes.misc import utils
from training_processes.misc.replay_buffer import ReplayBuffer
from training_processes.misc.lamb import Lamb
from pathlib import Path
from training_processes.misc.data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from training_processes.misc.evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from training_processes.misc.trainer import SequenceTrainer
from training_processes.misc.logger import Logger


MAX_EPISODE_LEN = 1000

class Experiment:
    def __init__(self, variant, environment, chargers, routes, state_dim, action_dim, device, seed, experiment_number, agg_num, zone_index, buffers, metrics_path, ev_info, date, verbose):

        self.persist_buffers = variant["persist_buffers"]
        self.metrics_base_path = metrics_path
        self.variant = variant
        self.experiment_number = experiment_number
        self.ev_info = ev_info
        self.date = date
        self.verbose = verbose
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.environment = environment
        self.chargers = chargers
        self.routes = routes
        self.seed = seed
        self.zone_index = zone_index
        self.agg_num = agg_num
        self.logger = Logger(variant, agg_num, zone_index)
        self.action_range = self._get_env_spec(variant)
        
        save_path = os.path.join(variant["save_dir"], variant["exp_name"])
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        
        if agg_num < 1 or not self.persist_buffers:
            self.offline_trajs, self.state_mean, self.state_std = self._load_dataset('merl')
            # Initialize by offline trajectories
            self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)
        
            # Save state_mean and state_std
            with open(os.path.join(save_path, 'state_stats.pkl'), 'wb') as f:
                pickle.dump({'state_mean': self.state_mean, 'state_std': self.state_std}, f)
        else:
            self.replay_buffer = buffers
            buffers = None
            # Load state_mean and state_std
            with open(os.path.join(save_path, 'state_stats.pkl'), 'rb') as f:
                state_stats = pickle.load(f)
            self.state_mean = state_stats['state_mean']
            self.state_std = state_stats['state_std']


        self.aug_trajs = []
        self.metrics = []
        self.device = device
        self.target_entropy = -self.act_dim
        
    
        self.model = DecisionTransformer(
        state_dim=self.state_dim,
        act_dim=self.act_dim,
        action_range=self.action_range,
        max_length=variant["K"],
        eval_context_length=variant["eval_context_length"],
        max_ep_len=MAX_EPISODE_LEN,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=variant["dropout"],
        stochastic_policy=True,
        ordering=variant["ordering"],
        init_temperature=variant["init_temperature"],
        target_entropy=self.target_entropy,
    ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        if agg_num > 0:
            start_time = time.time() 
            
            path = self.logger.log_path
            current_agg_num = int(re.search(r'Agg:(\d+)', path ).group(1))
            new_agg_num = current_agg_num - 1
            updated_save_model_path = re.sub(r'Agg:\d+', f'Agg:{new_agg_num}', path)
            self._load_model(updated_save_model_path)

            #Load aggregated weights
            save_global_path = f'saved_networks/Exp_{experiment_number}/'
            attn_layers = load_global_weights(save_global_path)
            self.set_attn_layers(self.model, attn_layers.to(device))
            end_time = time.time()
            elapsed = end_time - start_time
            print(f'Loading aggregated models took {elapsed}')
            


        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        
        self.reward_scale = 1
        

    def _get_env_spec(self, variant):
        action_range = [
            1e-6,
            1e-6,
        ]
        return action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")
        self.model_path = f"{path_prefix}/model.pt"

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):
        
        dataset_path = f"../Datasets/[{self.seed}]-{self.variant['offline_alg']}.pkl"
        if not os.path.exists(dataset_path):
            dataset_path = f"/mnt/storage_1/merl/[{self.seed}]-DQN.pkl"
            #dataset_path = f"/mnt/storage_1/merl/[1234]-Test.pkl"
            
        print('Loading Dataset...')
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
    
        states, traj_lens, returns = [], [], []
        
        for path in trajectories:
            # Convert lists to NumPy arrays to ensure compatibility
            path["observations"] = np.array(path["observations"], dtype=np.float32)#save memory
            path["rewards"] = np.array(path["rewards"])  # Convert rewards to NumPy array
            path["actions"] = np.array(path["actions"])  # If actions exist and are needed
    
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(sum(path['rewards']))
        
        traj_lens, returns = np.array(traj_lens), np.array(returns)
    
        # used for input normalization
        states = np.concatenate(states, axis=0)        
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)
    
        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)
    
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]
    
        return trajectories, state_mean, state_std


    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        online_iter,
        n,
        randomized=False
    ):
        max_ep_len = MAX_EPISODE_LEN
    
        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * 1
    
            returns, lengths, trajs, metrics = vec_evaluate_episode_rtg(
                online_envs,
                self.chargers,
                self.routes,
                self.state_dim,
                self.act_dim,
                self.environment.num_cars,
                self.zone_index,
                online_iter,
                self.agg_num,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
            )
        
        # Call evaluate() only every 5 iterations
        if online_iter % 5 == 0:
            directory = f"{self.metrics_base_path}/train/metrics"
            os.makedirs(directory, exist_ok=True)
    
            chunk_size = 50  # Define a reasonable chunk size
            for i, start in enumerate(range(0, len(metrics), chunk_size)):
                chunk = metrics[start:start + chunk_size]
                
                # Append is True for all chunks except the first one when aggregation_num > 0
                # append = aggregation_num > 0 or i > 0
    
                # Call evaluate with the current chunk
                evaluate(self.ev_info, chunk, self.seed, self.date, self.verbose, 'save', self.variant["max_online_iters"], directory, True, True)
    
        # Add trajectories to replay buffer
        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)
    
        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }


    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                chargers=self.chargers,
                routes=self.routes,
                num_cars=self.environment.num_cars,
                zone_index=self.zone_index,
                episode_num=self.pretrain_iter,
                aggregation_num=self.agg_num,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward
        
    def save_attn_layers(self, model, device):
        # Get the number of attention layers and their dimensions
        attn_number = len(model.transformer.h)
        attn_layer_size = model.transformer.h[0].attn.c_attn.weight.shape
        
        # Extract attention layers into a tensor
        attn_layers = torch.empty((attn_number, attn_layer_size[0], attn_layer_size[1]), dtype=torch.float32, device=self.device)
        for i in range(attn_number):
            attn_layers[i, :, :] = model.transformer.h[i].attn.c_attn.weight
        self.attn_layers = attn_layers

    def set_attn_layers(self, model, attn_layers):
        attn_number = len(model.transformer.h)
        for i in range(attn_number):
            model.transformer.h[i].attn.c_attn.weight = torch.nn.Parameter(attn_layers[i])
    
        return model
            
    def get_model_weights(self):
        return self.attn_layers

    def online_tuning(self, online_envs, eval_envs, loss_fn):
        print("\n\n\n*** Online Finetuning ***")
    
        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                chargers=self.chargers,
                routes=self.routes,
                num_cars=self.environment.num_cars,
                zone_index=self.zone_index,
                episode_num=self.online_iter,
                aggregation_num=self.agg_num,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
    
        buffer_to_return = None  # Initialize as None
        while self.online_iter < self.variant["max_online_iters"]:
            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                self.online_iter,
                n=self.variant["num_online_rollouts"],
                
            )
            outputs.update(augment_outputs)
    
            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )
    
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant["eval_interval"] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False
    
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)
    
            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)
    
            outputs["time/total"] = time.time() - self.start_time
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )
    
            if is_last_iter:
                self.save_attn_layers(self.model, self.device)
                if self.persist_buffers:
                    buffer_to_return = self.replay_buffer  # Assign buffer only at the last iteration
                else:
                    buffer_to_return = None
                
            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False
            )
            self.online_iter += 1 
    
        return buffer_to_return  # Return the buffer only at the last iteration


    def __call__(self):

        utils.set_seed_everywhere(self.seed)

        #import d4rl

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                env = self.environment
                return env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = 'merl'
        if "antmaze" in env_name:
            #env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
            eval_envs = self.environment

        self.start_time = time.time()
        if self.agg_num < 1 and self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = self.environment
            self.final_buffer = self.online_tuning(online_envs, eval_envs, loss_fn)
            
        #eval_envs.close()

def train_odt(ev_info, metrics_base_path, experiment_number, chargers, environment, routes, date, action_dim, global_weights, aggregation_num, zone_index, seed, main_seed, device, agent_by_zone, variant, args, fixed_attributes=None, verbose=False, display_training_times=False, 
              dtype=torch.float32, save_offline_data=False, train_model=True, old_buffers=None):

    utils.set_seed_everywhere(main_seed)
    experiment = Experiment(variant, environment, chargers, routes, 24, action_dim, device, main_seed, experiment_number, aggregation_num, zone_index, old_buffers, metrics_base_path, ev_info, date, verbose)

    print("=" * 50)
    experiment()
    
    metrics = experiment.metrics


    return experiment.get_model_weights().detach().cpu(), [], [], metrics, [], experiment.final_buffer



def load_global_weights(save_global_path):
    """
    Function to load global weights from the specified path and convert to a dictionary if necessary.

    Parameters:
    - save_global_path (str): Path to the saved global weights.

    Returns:
    - global_weights (dict): Loaded global weights, converted to a dictionary if originally a list.
    """
    weights_path = f'{save_global_path}/global_weights.pth'
    if Path(weights_path).exists():
        global_weights = torch.load(weights_path)

        # Check if global_weights is a list and convert to dictionary if needed
        if isinstance(global_weights, list):
            # Example conversion logic (adjust as per your needs)
            global_weights_dict = {}
            for i, weights_dict in enumerate(global_weights):
                if isinstance(weights_dict, dict):
                    for key, value in weights_dict.items():
                        # Create a unique key name if necessary
                        global_weights_dict[f"zone_{i}_{key}"] = value
                else:
                    raise ValueError("Expected a list of dictionaries, but found a different structure.")
            global_weights = global_weights_dict

            #print(f'Global_weights shape: {len(global_weights)}')
        return global_weights
    else:
        raise FileNotFoundError(f"No global weights found at {weights_path}")



