import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from gym.spaces.discrete import Discrete
# from sc2.toy_data import toy_example


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample(model, critic_model, state, obs, sample=False, actions=None, rtgs=None,
           timesteps=None, available_actions=None):
    if torch.cuda.is_available():
        block_size = model.module.get_block_size()
    else:
        block_size = model.get_block_size()
    model.eval()
    critic_model.eval()

    # Process inputs based on block size
    obs_cond = obs if obs.size(1) <= block_size//3 else obs[:, -block_size//3:]
    state_cond = state if state.size(1) <= block_size//3 else state[:, -block_size//3:]
    if actions is not None:
        actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:]
    rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:]
    timesteps = timesteps if timesteps.size(1) <= block_size//3 else timesteps[:, -block_size//3:]

    action_mean, action_std = model(obs_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps)
    
   
    
    # Check for NaNs in action_mean and action_std

    
    # Sample the action
    if sample:
        # Sample from the Gaussian distribution
        a = action_mean + action_std * torch.randn_like(action_std)
    else:
        # Just use the mean for deterministic action
        a = action_mean
        

    # Value prediction using the critic
    v = critic_model(states=state_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps).detach()
    v = v[:, -1, :]
    
    return a, v


def get_dim_from_space(space):
    if isinstance(space[0], Discrete):
        return space[0].n
    elif isinstance(space[0], list):
        return space[0][0]


def padding_obs(obs, target_dim):
    len_obs = np.shape(obs)[-1]
    if len_obs > target_dim:
        print("target_dim (%s) too small, obs dim is %s." % (target_dim, len(obs)))
        raise NotImplementedError
    elif len_obs < target_dim:
        padding_size = target_dim - len_obs
        if isinstance(obs, list):
            obs = np.array(copy.deepcopy(obs))
            padding = np.zeros(padding_size)
            obs = np.concatenate((obs, padding), axis=-1).tolist()
        elif isinstance(obs, np.ndarray):
            obs = copy.deepcopy(obs)
            shape = np.shape(obs)
            padding = np.zeros((shape[0], shape[1], padding_size))
            obs = np.concatenate((obs, padding), axis=-1)
        else:
            print("unknwon type %s." % type(obs))
            raise NotImplementedError
    return obs


def padding_ava(ava, target_dim):
    len_ava = np.shape(ava)[-1]
    if len_ava > target_dim:
        print("target_dim (%s) too small, ava dim is %s." % (target_dim, len(ava)))
        raise NotImplementedError
    elif len_ava < target_dim:
        padding_size = target_dim - len_ava
        if isinstance(ava, list):
            ava = np.array(copy.deepcopy(ava), dtype=np.long)
            padding = np.zeros(padding_size, dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1).tolist()
        elif isinstance(ava, np.ndarray):
            ava = copy.deepcopy(ava)
            shape = np.shape(ava)
            padding = np.zeros((shape[0], shape[1], padding_size), dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1)
        else:
            print("unknwon type %s." % type(ava))
            raise NotImplementedError
    return ava
