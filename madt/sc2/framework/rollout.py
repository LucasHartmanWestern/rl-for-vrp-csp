import copy

import torch
import numpy as np
from .utils import sample, padding_obs, padding_ava


class RolloutWorker:

    def __init__(self, model, critic_model, buffer, global_obs_dim, local_obs_dim, action_dim):
        self.buffer = buffer
        self.model = model
        self.critic_model = critic_model
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.device = 'cpu'
        if torch.cuda.is_available() and not isinstance(self.model, torch.nn.DataParallel):
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(model).to(self.device)
            self.critic_model = torch.nn.DataParallel(critic_model).to(self.device)

    def rollout(self, env, ret, train=True, random_rate=0.):
        self.model.train(False)
        self.critic_model.train(False)

        T_rewards, T_wins, steps, episode_dones = 0., 0., 0, np.zeros(env.n_threads)

        obs, share_obs, available_actions = env.real_env.reset()
        obs = padding_obs(obs, self.local_obs_dim)
        share_obs = padding_obs(share_obs, self.global_obs_dim)
        available_actions = None #padding_ava(available_actions, self.action_dim)

        # x: (n_threads, n_agent, context_lengrh, dim)
        local_obss = torch.from_numpy(obs).to(self.device).unsqueeze(2)
        global_states = torch.from_numpy(share_obs).to(self.device).unsqueeze(1).repeat(1, local_obss.size(1), 1, 1)
        #global_states = global_states.expand(-local_obss.size(1), -1, -1)
        rtgs = np.ones((env.n_threads, env.num_agents, 1, 1)) * ret
        actions = np.zeros((env.n_threads, env.num_agents, 1, 9))  # Initialize with dimension 9 instead of 1
        timesteps = torch.zeros((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
        t = 0

        while True:
            sampled_action, v_value = sample(
                self.model, 
                self.critic_model, 
                state = global_states.view(-1, global_states.size(2), global_states.size(3)),
                obs = local_obss.view(-1, local_obss.size(2), local_obss.size(3)), 
                sample=train,
                actions=torch.tensor(actions, dtype=torch.int64).to(self.device).view(-1, np.shape(actions)[2], np.shape(actions)[3]),
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).view(-1, np.shape(rtgs)[2], np.shape(rtgs)[3]),
                timesteps=timesteps.to(self.device),
                available_actions=None
            ) #torch.from_numpy(available_actions).view(-1, np.shape(available_actions)[-1]))
            
            action = sampled_action.detach().view((env.n_threads, env.num_agents, -1)).cpu().numpy()

            cur_global_obs = share_obs
            cur_local_obs = obs
            cur_ava = available_actions

            obs, share_obs, rewards, dones, infos, available_actions = env.real_env.step(action)
            obs = padding_obs(obs, self.local_obs_dim)
            share_obs = padding_obs(share_obs, self.global_obs_dim)
           # available_actions = padding_ava(available_actions, self.action_dim)
            t += 1

            if train:
                expected_size = env.n_threads * env.num_agents
                if v_value.shape[0] > expected_size:
                    #print(f"Truncating v_value from {v_value.shape[0]} to {expected_size}")
                    v_value = v_value[:expected_size]
                v_value = v_value.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
               # print(f"v_value shape after reshape: {v_value.shape}")
                self.buffer.insert(cur_global_obs, cur_local_obs, action, rewards, dones, cur_ava, v_value)

            for n in range(env.n_threads):
                if not episode_dones[n]:
                    steps += 1
                    T_rewards += np.mean(rewards[n])
                    if np.all(dones[n]):
                        episode_dones[n] = 1
                        # if infos[n][0]['won']:
                        #     T_wins += 1.
            if np.all(episode_dones):
                break

            rtgs = np.concatenate([rtgs, np.expand_dims(rtgs[:, :, -1, :] - rewards, axis=2)], axis=2)
            #global_state = torch.from_numpy(share_obs).to(self.device).unsqueeze(0).unsqueeze(2)  # Add necessary dimensions
            # Adjust global_state to match the dimensions of global_states
            global_state = torch.from_numpy(share_obs).to(self.device).unsqueeze(0).unsqueeze(2)  # (1, num_agents, 1, 2)
            global_state = global_state.expand(-1, -1, global_states.size(2), -1)  # Match current timestep dimension
            
            # Now they have matching dimensions for concatenation
            global_states = torch.cat([global_states, global_state], dim=1)
            local_obs = torch.from_numpy(obs).to(self.device).unsqueeze(2)
            local_obss = torch.cat([local_obss, local_obs], dim=2)
            timestep = t * torch.ones((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
            timesteps = torch.cat([timesteps, timestep], dim=1)

        aver_return = T_rewards / env.n_threads
        # aver_win_rate = T_wins / env.n_threads
        self.model.train(True)
        self.critic_model.train(True)
        return aver_return, steps #aver_return, aver_win_rate, steps
