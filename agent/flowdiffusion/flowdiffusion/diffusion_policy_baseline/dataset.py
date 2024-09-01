import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import pathlib
import torch
from einops import rearrange


def compute_mean_std(states: np.ndarray, eps: float):
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

class MWDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        output_dir,
        horizon_length = 4
    ):
        super().__init__()
        self.horizon_length = horizon_length
        data_directory = pathlib.Path(dataset_dir)
        print("Loading dataset from", data_directory)
        with open(data_directory, 'rb') as f:
            data = pkl.load(f)
        observations = data['observations']
        actions = data['actions']
        terminals = data['terminals']
        images = data['images']
        print("Images shape: ", images[0].shape)
        self.obs_mean, self.obs_std = compute_mean_std(observations, 1e-3) 
        observations = normalize_states(observations, self.obs_mean, self.obs_std)
        
        self.actions_mean, self.actions_std = compute_mean_std(actions, 1e-3)
        actions = normalize_states(actions, self.actions_mean, self.actions_std)
        
        data_points = []
        output_dir = pathlib.Path(output_dir)
        print("Saving dataset to", output_dir)
        with open(output_dir / "data_stats.pkl", 'wb') as f:
            pkl.dump([self.obs_mean, self.obs_std, self.actions_mean, self.actions_std], f)
        end_ = np.where(terminals == 1)
        for ind in end_:
            i = self.horizon_length 
            while i >= 0:
                terminals[ind - i] = 1
                i =  i - 1 
                
                
        self.observations = []
        self.actions =  []
        self.images = []
        
        for i in range(len(images)):
            if terminals[i] == 1:
                continue
            # print(images[i].shape)
            # exit()
            # images[i] = rearrange(images[i], 'h w c -> c h w')
            # images[i+1] = rearrange(images[i+1], 'h w c -> c h w')
            self.images.append(torch.Tensor(np.array([images[i], images[i+1]])).permute(0, 3, 1, 2))
            self.actions.append(torch.Tensor(actions[i + 1 : i + self.horizon_length + 1]))
            self.observations.append(torch.Tensor(np.array([observations[i], observations[i+1]])))
            
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {}
        data['image'] = self.images[idx]
        data['action'] = self.actions[idx]
        data['observation'] = self.observations[idx]        
        return data