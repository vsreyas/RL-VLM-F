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
import os


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
        horizon_length=4
    ):
        super().__init__()
        self.horizon_length = horizon_length
        self.dataset_dir = dataset_dir
        data_directory = pathlib.Path(dataset_dir)
        print("Loading dataset from", data_directory)
        
        data_dict = {"observations": [], "actions": [], "images_path": [], "terminals": []}
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(dataset_dir, file_name)
                print(f'Loading data from {file_path}')
                with open(file_path, 'rb') as file:
                    data = pkl.load(file)
                    data_dict["observations"].append(data["observations"])
                    data_dict["actions"].append(data["actions"])
                    data_dict["images_path"].append(data["images_path"])
                    data_dict["terminals"].append(data["terminals"])
                    del data
                    
        # Concatenate lists
        data_dict["observations"] = np.concatenate(data_dict["observations"])
        data_dict["actions"] = np.concatenate(data_dict["actions"])
        data_dict["terminals"] = np.concatenate(data_dict["terminals"])
        data_dict["images_path"] = np.concatenate(data_dict["images_path"])       
       
        observations = data_dict['observations']
        actions = data_dict['actions']
        terminals = data_dict['terminals']
        images_path = data_dict['images_path']
        
        print("Images shape: ", images_path.shape)
        self.obs_mean, self.obs_std = compute_mean_std(observations, 1e-3) 
        observations = normalize_states(observations, self.obs_mean, self.obs_std)
        
        self.actions_mean, self.actions_std = compute_mean_std(actions, 1e-3)
        actions = normalize_states(actions, self.actions_mean, self.actions_std)
        
        # Save dataset stats
        output_dir = pathlib.Path(output_dir)
        print("Saving dataset to", output_dir)
        with open(output_dir / "data_stats.pkl", 'wb') as f:
            pkl.dump([self.obs_mean, self.obs_std, self.actions_mean, self.actions_std], f)
        
        # Adjust terminal states
        end_ = np.where(terminals == 1)[0]
        for ind in end_:
            i = self.horizon_length + 1
            while i >= 0:
                if ind - i >= 0:
                    terminals[ind - i] = 1
                i -= 1 
        ind = len(images_path) - 1
        i = self.horizon_length + 1
        while i >= 0:
            if ind - i >= 0:
                terminals[ind - i] = 1
            i -= 1         
        self.observations = []
        self.actions =  []
        self.images_path = []
        
        for i in range(len(images_path) - 1):
            if terminals[i] == 1:
                continue
            abs_image_path1 = os.path.join(data_directory, images_path[i])
            abs_image_path2 = os.path.join(data_directory, images_path[i + 1])
            
            self.images_path.append([abs_image_path1, abs_image_path2])
            self.actions.append(np.array(actions[i + 1 : i + self.horizon_length + 1]))
            self.observations.append(np.array([observations[i], observations[i + 1]]))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        try:
            # Debugging output to track the index
            
            img1 = np.load(self.images_path[idx][0])
            img2 = np.load(self.images_path[idx][1])

            # Check if the image shapes are consistent
            if img1.shape != img2.shape:
                raise RuntimeError(f"Inconsistent image shapes at index {idx}: {img1.shape}, {img2.shape}")

            imgs = np.array([img1, img2])
            data = {}

            # Permute image dimensions to [batch_size, channels, height, width]
            data['image'] = torch.Tensor(imgs).permute(0, 3, 1, 2)

            # Check actions and observations shape
            actions = self.actions[idx]
            observations = self.observations[idx]
            
            if actions.shape[0] != self.horizon_length:
                raise RuntimeError(f"Inconsistent action length at index {idx}: {actions.shape}")
            
            if observations.shape[0] != 2:
                raise RuntimeError(f"Inconsistent observation length at index {idx}: {observations.shape}")

            # Add actions and observations to the data dict
            data['action'] = torch.Tensor(actions)
            data['observation'] = torch.Tensor(observations)

            # Print action and observation shapes for debugging
            
            return data
        except Exception as e:
            # Print exception details to help identify the issue
            print(f"Error at index {idx}: {str(e)}")
            print(f"Processing index: {idx}")
            
            # Load images and check shapes
            print(f"Loading image paths: {self.images_path[idx]}")
            print(f"Loaded images shapes: {img1.shape}, {img2.shape}")
            print(f"Actions shape: {data['action'].shape}, Observations shape: {data['observation'].shape}")


            raise e
