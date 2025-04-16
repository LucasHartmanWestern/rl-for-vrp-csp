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
import h5py
import os


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

# For offline saving
def format_data(data):
    # Flatten the data if it's a list of lists
    if isinstance(data, list) and all(isinstance(sublist, list) for sublist in data):
        flattened_data = [item for sublist in data for item in sublist]
    elif isinstance(data, list):
        flattened_data = data
    else:
        raise TypeError("Input data must be a list or a list of lists.")

    # Initialize defaultdict for organized aggregation
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

    # Iterate over data entries to aggregate
    for entry in flattened_data:
        if not isinstance(entry, dict):
            raise TypeError(f"Entry is not a dictionary: {entry}")

        identifier = (entry['zone'], entry['aggregation'], entry['episode'], entry['car_idx'])

        # Consistency check for metadata values
        for meta_key in ['zone', 'aggregation', 'episode', 'car_idx']:
            if trajectories[identifier][meta_key] is not None and trajectories[identifier][meta_key] != entry[meta_key]:
                raise ValueError(f"Inconsistent {meta_key} value for identifier {identifier}")

        # Efficient aggregation using `.append()` followed by flattening
        for key in ['observations', 'actions', 'rewards', 'terminals', 'terminals_car']:
            trajectories[identifier][key].append(entry.get(key, []))

        # Store metadata only once
        for meta_key in ['zone', 'aggregation', 'episode', 'car_idx']:
            trajectories[identifier][meta_key] = entry.get(meta_key)

    # Flatten the appended lists for improved memory efficiency
    for traj in trajectories.values():
        for key in ['observations', 'actions', 'rewards', 'terminals', 'terminals_car']:
            traj[key] = [item for sublist in traj[key] for item in sublist]

    # Convert defaultdict to list of dictionaries
    formatted_trajectories = list(trajectories.values())

    return formatted_trajectories

def save_to_h5(data, path, zone_index):
    """Saves RL data structured as a list of dictionaries into `.h5` format."""
    temp_path = path + ".tmp"

    with h5py.File(temp_path, 'w') as f:
        zone_grp = f.create_group(f"zone_{zone_index}")

        for i, entry in enumerate(data):
            traj_grp = zone_grp.create_group(f"traj_{i}")

            for key, value in entry.items():
                if isinstance(value, (list, np.ndarray)):
                    traj_grp.create_dataset(key, data=np.array(value))
                else:
                    traj_grp.attrs[key] = value

    #Ensures incomplete writes don't corrupt the data
    os.replace(temp_path, path)
    print(f"Data for Zone {zone_index} saved successfully to {path}")

def save_temp_checkpoint(data, path, zone_index):
    """Write data to a temporary .h5 checkpoint file and return the filename."""
    temp_path = path + ".tmp.h5"
    with h5py.File(temp_path, 'w') as f:
        zone_grp = f.create_group(f"zone_{zone_index}")
        for i, entry in enumerate(data):
            traj_grp = zone_grp.create_group(f"traj_{i}")
            for key, value in entry.items():
                traj_grp.create_dataset(key, data=np.array(value) if isinstance(value, (list, np.ndarray)) else value)
    return temp_path

