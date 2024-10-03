import torch
import numpy as np
import pickle
import csv
from collections import defaultdict
import copy, glob
from torch.utils.data import Dataset
from .utils import padding_obs, padding_ava


class StateActionReturnDataset(Dataset):

    def __init__(self, global_state, local_obs, block_size, actions, done_idxs, rewards, avas, v_values, rtgs, rets,
                 advs, timesteps):
        self.block_size = block_size
        self.global_state = global_state
        self.local_obs = local_obs
        self.actions = actions
        self.done_idxs = done_idxs
        self.rewards = rewards
        self.avas = avas
        self.v_values = v_values
        self.rtgs = rtgs
        self.rets = rets
        self.advs = advs
        self.timesteps = timesteps

    def __len__(self):
        # return len(self.global_state) - self.block_size
        return len(self.global_state)

    def stats(self):
        print("max episode length: ", max(np.array(self.done_idxs[1:]) - np.array(self.done_idxs[:-1])))
        print("min episode length: ", min(np.array(self.done_idxs[1:]) - np.array(self.done_idxs[:-1])))
        print("max rtgs: ", max(self.rtgs))
        print("aver episode rtgs: ", np.mean([self.rtgs[i] for i in self.done_idxs[:-1]]))

    @property
    def max_rtgs(self):
        return max(self.rtgs)[0]

    def __getitem__(self, idx):
        context_length = self.block_size // 3
        done_idx = idx + context_length
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - context_length
        states = torch.tensor(np.array(self.global_state[idx:done_idx]), dtype=torch.float32)
        
        obss = torch.tensor(np.array(self.local_obs[idx:done_idx]), dtype=torch.float32)
    
        if done_idx in self.done_idxs:
            next_states = [np.zeros_like(self.global_state[idx]).tolist()] + self.global_state[idx+1:done_idx] + \
                          [np.zeros_like(self.global_state[idx]).tolist()]
            next_states.pop(0)
            next_rtgs = [np.zeros_like(self.rtgs[idx]).tolist()] + self.rtgs[idx+1:done_idx] + \
                        [np.zeros_like(self.rtgs[idx]).tolist()]
            next_rtgs.pop(0)
        else:
            next_states = self.global_state[idx+1:done_idx+1]
            next_rtgs = self.rtgs[idx+1:done_idx+1]
        next_states = torch.tensor(next_states, dtype=torch.float32)
        next_rtgs = torch.tensor(next_rtgs, dtype=torch.float32)
    
        if idx == 0 or idx in self.done_idxs:
            pre_actions = [[0] * 9] + self.actions[idx:done_idx-1]
        else:
            pre_actions = self.actions[idx-1:done_idx-1]
    
        avas = self.avas[idx:done_idx]
        if avas is None or all(a is None for a in avas):  # If avas is None or all elements are None
            avas = torch.zeros(done_idx - idx, 9, dtype=torch.long)  # Assuming 9 is the action dimension
        else:
            avas = torch.tensor(avas, dtype=torch.long)

        # Check and handle NaNs for pre_actions
        if np.isnan(pre_actions).any():
            pre_actions = np.nan_to_num(pre_actions, nan=0)  # Replace NaNs with 0
        pre_actions = torch.tensor(pre_actions, dtype=torch.long)
        
        # Check and handle NaNs for actions
        actions_slice = self.actions[idx:done_idx]  # Apply the slice only once
        if np.isnan(actions_slice).any():
            actions_slice = np.nan_to_num(actions_slice, nan=0)
        actions = torch.tensor(actions_slice, dtype=torch.long)
        
        # Check and handle NaNs for rewards
        rewards_slice = self.rewards[idx:done_idx]
        if np.isnan(rewards_slice).any():
            rewards_slice = np.nan_to_num(rewards_slice, nan=0) + 0.000001
        rewards = torch.tensor(rewards_slice, dtype=torch.float32)
        
        # Check and handle NaNs for v_values
        v_values_slice = self.v_values[idx:done_idx]
        if np.isnan(v_values_slice).any():
            v_values_slice = np.nan_to_num(v_values_slice, nan=0) + 0.000001
        v_values = torch.tensor(v_values_slice, dtype=torch.float32)
        
        # Check and handle NaNs for rtgs
        rtgs_slice = self.rtgs[idx:done_idx]
        if np.isnan(rtgs_slice).any():
            rtgs_slice = np.nan_to_num(rtgs_slice, nan=0) + 0.000001
        rtgs = torch.tensor(rtgs_slice, dtype=torch.float32)
        
        # Check and handle NaNs for rets
        rets_slice = self.rets[idx:done_idx]
        if np.isnan(rets_slice).any():
            rets_slice = np.nan_to_num(rets_slice, nan=0) + 0.000001
        rets = torch.tensor(rets_slice, dtype=torch.float32)
        
        # Check and handle NaNs for advs
        advs_slice = self.advs[idx:done_idx]
        if np.isnan(advs_slice).any():
            advs_slice = np.nan_to_num(advs_slice, nan=0) + 0.000001
        advs = torch.tensor(advs_slice, dtype=torch.float32)
        
        # Check and handle NaNs for timesteps
        timesteps_slice = self.timesteps[idx:done_idx]
        if np.isnan(timesteps_slice).any():
            timesteps_slice = np.nan_to_num(timesteps_slice, nan=0) + 0.000001
        timesteps = torch.tensor(timesteps_slice, dtype=torch.int64)
    
        dones = torch.zeros_like(rewards)
        if done_idx in self.done_idxs:
            dones[-1] = 1
        
        return states, obss, actions, rewards, avas, v_values, rtgs, rets, advs, timesteps, pre_actions, next_states, next_rtgs, dones

class ReplayBuffer:

    def __init__(self, block_size, global_obs_dim, local_obs_dim, action_dim):
        self.block_size = block_size
        self.buffer_size = 5000
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.data = []
        self.episodes = []
        self.episode_dones = []
        self.gamma = 0.99
        self.gae_lambda = 0.95

    @property
    def size(self):
        return len(self.data)

    def insert(self, global_obs, local_obs, action, reward, done, available_actions, v_value):
        n_threads, n_agents = np.shape(reward)[0], np.shape(reward)[1]
        #global_obs = torch.from_numpy(global_obs).to(self.device).expand(-1, env.num_agents, -1, -1)
        
        done = np.transpose(done, (0, 2, 1))
        
        for n in range(n_threads):
            if len(self.episodes) < n + 1:
                self.episodes.append([])
                self.episode_dones.append(False)
            if not self.episode_dones[n]:
                global_obs_for_agents = global_obs[n].tolist()
                for i in range(n_agents):
                    if len(self.episodes[n]) < i + 1:
                        self.episodes[n].append([])
                    
                    step = [global_obs_for_agents, local_obs[n][i].tolist(), action[n][i].tolist(),
                            reward[n][i].tolist(), done[n][i], None, v_value[n][i].tolist()]
                    self.episodes[n][i].append(step)
                    
                if np.all(done[n]):
                    self.episode_dones[n] = True
                    if self.size > self.buffer_size:
                        raise NotImplementedError
                    if self.size == self.buffer_size:
                        del self.data[0]
                    self.data.append(copy.deepcopy(self.episodes[n]))
        
        if np.all(self.episode_dones):
            self.episodes = []
            self.episode_dones = []


    def reset(self, num_keep=0, buffer_size=5000):
        self.buffer_size = buffer_size
        if num_keep == 0:
            self.data = []
        elif self.size >= num_keep:
            keep_idx = np.random.randint(0, self.size, num_keep)
            self.data = [self.data[idx] for idx in keep_idx]

    def process_merl_data(self, file_path):
        file_path = file_path + '[1234]-10-3-2-10-20240830_120849.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Append available actions as None (if not already included in the data)
        for entry in data:
            if 'available_actions' not in entry:
                entry['available_actions'] = None
                
        episodes = defaultdict(lambda: defaultdict(list))
        
        # Group trajectories by aggregation and episode number
        for trajectory in data:
            aggregation_num = trajectory['aggregation']
            episode_num = trajectory['episode']
            episodes[aggregation_num][episode_num].append(trajectory)
    
        processed_data = []
    
        # Process each episode within each aggregation
        for aggregation_num, episodes_in_aggregation in episodes.items():
            for episode_num, trajectories in episodes_in_aggregation.items():
                episode = []  # List of trajectories for the episode
                
                for trajectory in trajectories:
                    trajectory_steps = []
                    num_steps = len(trajectory['observations'])
                    
                    for step_idx in range(num_steps):
                        state = trajectory['observations'][step_idx]
                        action = trajectory['actions'][step_idx]
                        reward = trajectory['rewards'][step_idx]
                        terminal_by_car = trajectory['terminals_car'][step_idx]
                        available_actions = trajectory['available_actions']
                        
                        # Split state into global and local components
                        global_states = []
                        local_obss = []
                        
                        for i, element in enumerate(state):
                            if i == 18 or i == 20:
                                global_states.append(element)
                            else:
                                local_obss.append(element)
                        
                        # Create a single step as a list
                        step = [
                            global_states,        # Global states
                            local_obss,           # Local observations
                            action,               # Action
                            [reward],             # Reward
                            terminal_by_car,      # Terminal flag
                            available_actions     # Available actions
                        ]
                        
                        trajectory_steps.append(step)  # Append the step to the trajectory
                    
                    episode.append(trajectory_steps)  # Append the full trajectory (all steps) to the episode
                
                processed_data.append(episode)  # Append the entire episode to the processed data
        
        return processed_data

    # offline data size could be large than buffer size
    def load_offline_data(self, data_dir, offline_episode_num, max_epi_length=400):
        output_csv_path = 'processed_data2.csv'
        
        for j in range(1):        
            if data_dir == '/storage_1/epigou_storage/madt/merl_data/':
                # Process MERL data if the directory matches the specific path
                episodes_data = self.process_merl_data(data_dir)
            else:
                # Otherwise, load data using glob
                path_files = glob.glob(pathname=data_dir[j] + "*")
                episodes_data = []
                
                for i in range(offline_episode_num[j]):
                    episode = torch.load(path_files[i])
                    
                    # Optional: Filter episodes by max episode length (if required)
                    if len(episode[0]) > max_epi_length:
                        print(f"Skipping episode {i} in {data_dir[j]}: Exceeds max length")
                        continue
                    
                    episodes_data.append(episode)
    
            # Iterate over the episodes and process them
            for episode in episodes_data:
                for trajectory in episode:
                    for step in trajectory:
                        step[0] = padding_obs(step[0], self.global_obs_dim)  # Global states
                        step[1] = padding_obs(step[1], self.local_obs_dim)   # Local observations
                        if step[5] is not None:
                            step[5] = padding_ava(step[5], self.action_dim)  # Available actions
                
                # Append the fully processed episode to self.data
                self.data.append(episode)
    
        # Write processed data to CSV
        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['global_states', 'local_obss', 'actions', 'rewards', 'terminals_by_car', 'available_actions'])
            
            # Iterate over episodes to write them into the CSV
            for episode in self.data:
                for trajectory in episode:
                    for step in trajectory:
                        csv_writer.writerow(step)
        

    def sample(self):
        # adding elements with list will be faster
        global_states = []
        local_obss = []
        actions = []
        rewards = []
        avas = []
        v_values = []
        rtgs = []
        rets = []
        done_idxs = []
        time_steps = []
        advs = []

        for episode_idx in range(self.size):
            episode = self.get_episode(episode_idx)
            # episode = self.get_episode(episode_idx, min_return)
            if episode is None:
                continue
            for agent_trajectory in episode:
                time_step = 0
                for step in agent_trajectory:
                    g, l, a, r, d, ava, v, rtg, ret, adv = step
                    global_states.append(g)
                    local_obss.append(l)
                    actions.append(a[:9])
                    rewards.append(r)
                    avas.append(ava)
                    v_values.append(v)
                    rtgs.append(rtg)
                    rets.append(ret)
                    advs.append(adv)
                    time_steps.append([time_step])
                    time_step += 1
                # done_idx - 1 equals the last step's position
                done_idxs.append(len(global_states))

        dataset = StateActionReturnDataset(global_states, local_obss, self.block_size, actions, done_idxs, rewards,
                                           avas, v_values, rtgs, rets, advs, time_steps)
        #print(f'online actions: {actions}')
        return dataset

    # from [g, o, a, r, d, ava]/[g, o, a, r, d, ava, v] to [g, o, a, r, d, ava, v, rtg, ret, adv]
    def get_episode(self, index):
        episode = copy.deepcopy(self.data[index])

        # cal rtg and ret
        for agent_trajectory in episode:
            rtg = 0.
            ret = 0.
            adv = 0.
            for i in reversed(range(len(agent_trajectory))):
                if len(agent_trajectory[i]) == 6:  # offline, give a fake v_value, unused
                    agent_trajectory[i].append([0.])
                elif len(agent_trajectory[i]) == 7:
                    pass  # online nothing to do
                else:
                    raise NotImplementedError
                reward = agent_trajectory[i][3][0]
                rtg += reward
                agent_trajectory[i].append([rtg])

                # todo: check ret and adv calculation
                if i == len(agent_trajectory) - 1:
                    next_v = 0.
                else:
                    next_v = agent_trajectory[i + 1][6][0]
                v = agent_trajectory[i][6][0]
                # adv with gae
                delta = reward + self.gamma * next_v - v
                adv = delta + self.gamma * self.gae_lambda * adv

                # adv without gae
                # adv = reward + self.gamma * next_v - v

                # ret = adv + v
                ret = reward + self.gamma * ret
                # ret = reward + self.gamma * next_v

                agent_trajectory[i].append([ret])
                agent_trajectory[i].append([adv])

        # prune dead steps
        for i in range(len(episode)):
            end_idx = 0
            for step in episode[i]:
                if step[4]:
                    break
                else:
                    end_idx += 1
            episode[i] = episode[i][0:end_idx + 1]
        return episode