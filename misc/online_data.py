import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from misc.data import TransformSamplingSubTraj, MAX_EPISODE_LEN


class PersistentOnlineDataset(Dataset):
    """
    A persistent dataset that maintains a fixed-size sample of trajectories.
    New online trajectories are added by removing the oldest ones in the sample.
    """
    def __init__(self, initial_trajectories, sample_size, transform):
        self.trajectories = list(initial_trajectories)
        self.sample_size = sample_size
        self.transform = transform

        # Initialize with the first N indices
        self.sampling_ind = list(range(min(len(self.trajectories), self.sample_size)))

    def update_with_new_trajectories(self, new_trajs):
        """
        Add new trajectories and maintain fixed sample size by removing the oldest indices.
        """
        start_idx = len(self.trajectories)
        self.trajectories.extend(new_trajs)

        num_new = len(new_trajs)

        # Remove oldest indices
        self.sampling_ind = self.sampling_ind[num_new:]
        # Add indices for new trajectories
        self.sampling_ind.extend(range(start_idx, start_idx + num_new))

        # Clip if we went over (in case of more new trajs than sample size)
        self.sampling_ind = self.sampling_ind[-self.sample_size:]

    def __len__(self):
        return len(self.sampling_ind)

    def __getitem__(self, index):
        traj = self.trajectories[self.sampling_ind[index]]
        return self.transform(traj)


def create_online_dataloader(dataset, batch_size, num_workers=0):
    """
    Standard DataLoader wrapper for the persistent dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
