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
        print("List of files in the directory", os.listdir(dataset_dir))
        
        data_dict = {"observations": [], "actions": [], "images": [], "terminals": []}
        for file_name in os.listdir(dataset_dir):
            if file_name.startswith('data_') and file_name.endswith('.0.pkl'):
                file_path = os.path.join(dataset_dir, file_name)
                print(f'Loading data from {file_path}')
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    print(data.keys())
                    print("Length:",len(data["observations"]))
                    # Append the data from each file into the corresponding lists in data_dict
                    data_dict["observations"].append(data["observations"])
                    data_dict["actions"].append(data["actions"])
                    data_dict["images"].append(data["images"])
                    data_dict["terminals"].append(data["terminals"])
                    del data
        data_dict["observations"] = np.concatenate(data_dict["observations"])
        data_dict["actions"] = np.concatenate(data_dict["actions"])
        data_dict["terminals"] = np.concatenate(data_dict["terminals"])
        data_dict["images"] = np.concatenate(data_dict["images"])       
        
        observations = data_dict['observations']
        actions = data_dict['actions']
        terminals = data_dict['terminals']
        images = data_dict['images']
        
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
            self.images.append(np.array([images[i], images[i+1]]))
            self.actions.append(actions[i + 1 : i + self.horizon_length + 1])
            self.observations.append(np.array([observations[i], observations[i+1]]))
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = {}
        data['image'] = torch.Tensor(self.images[idx]).permute(0, 3, 1, 2)
        data['action'] = torch.Tensor(self.actions[idx])
        data['observation'] = torch.Tensor(self.observations[idx])        
        return data