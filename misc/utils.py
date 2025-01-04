"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
import random
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_np(t):
    """
    convert a torch tensor to a numpy array
    """
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

#For offline saving
def format_data(data):
    # Flatten the data if it's a list of lists
    if isinstance(data, list) and all(isinstance(sublist, list) for sublist in data):
        flattened_data = [item for sublist in data for item in sublist]
    elif isinstance(data, list):
        flattened_data = data
    else:
        raise TypeError("Input data must be a list or a list of lists.")

    # Initialize a defaultdict to aggregate data by unique identifiers
    trajectories = defaultdict(lambda: {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
        'terminals_car': [],
        'zone': None,
        'aggregation': None,
        'episode': None,
        'car_idx': None
    })

    # Iterate over each data entry to aggregate the data
    for entry in flattened_data:
        if not isinstance(entry, dict):
            raise TypeError(f"Entry is not a dictionary: {entry}")
        # Unique identifier for each car's trajectory
        identifier = (entry['zone'], entry['aggregation'], entry['episode'], entry['car_idx'])

        # Aggregate data for this car's trajectory
        trajectories[identifier]['observations'].extend(entry.get('observations', []))
        trajectories[identifier]['actions'].extend(entry.get('actions', []))
        trajectories[identifier]['rewards'].extend(entry.get('rewards', []))
        trajectories[identifier]['terminals'].extend(entry.get('terminals', []))
        trajectories[identifier]['terminals_car'].extend(entry.get('terminals_car', []))
        trajectories[identifier]['zone'] = entry.get('zone')
        trajectories[identifier]['aggregation'] = entry.get('aggregation')
        trajectories[identifier]['episode'] = entry.get('episode')
        trajectories[identifier]['car_idx'] = entry.get('car_idx')

    # Convert the defaultdict to a list of dictionaries
    formatted_trajectories = list(trajectories.values())
    return formatted_trajectories