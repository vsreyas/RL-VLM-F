#!/usr/bin/env python3\
import sys
sys.path.append('/project_data/held/sreyas/RL-VLM-F')
import numpy as np
import torch
import os
import time
import pickle as pkl
import glob
from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from reward_model_score import RewardModelScore
from collections import deque
from prompt import clip_env_prompts
import copy
import utils
import hydra
from PIL import Image
from prompt import clip_env_prompts
import clip
from PIL import Image
from matplotlib import pyplot as plt
from config import Config
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)


import cv2

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from math import ceil
from tqdm import tqdm
import pyrallis
from dataclasses import dataclass,asdict
import os
import pickle
import multiprocessing as mp
import imageio
import inspect
import uuid

class DataGen(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")
        self.multiple = True
        self.model_paths = ["0-25","0-50","0-75", "prox_flip"]
        self.cfg = cfg
        self.cfg.prompt = clip_env_prompts[cfg.env]
        self.cfg.clip_prompt = clip_env_prompts[cfg.env]
        self.reward = self.cfg.reward # what types of reward to use
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        os.system("cp {}/prompt.py {}/".format(current_file_path, self.logger._log_dir))
        print("Number of trajs: ", cfg.num_eval_episodes)
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        elif cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            self.env = utils.make_classic_control_env(cfg)
        elif 'softgym' in cfg.env:
            self.env = utils.make_softgym_env(cfg)
        else:
            self.env = utils.make_env(cfg)
        
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        
        image_height = image_width = cfg.image_size
        self.resize_factor = 1
        if "sweep" in cfg.env or 'drawer' in cfg.env or "soccer" in cfg.env:
            print("Setting image size to 300 for {}".format(cfg.env))
            image_height = image_width = 300 
        if "Rope" in cfg.env:
            image_height = image_width = 240
            self.resize_factor = 3
        elif "Water" in cfg.env:
            image_height = image_width = 360
            self.resize_factor = 2
        if "CartPole" in cfg.env:
            image_height = image_width = 200
        if "Cloth" in cfg.env:
            image_height = image_width = 360
            
        self.image_height = image_height
        self.image_width = image_width

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity) if not self.cfg.image_reward else 200000, # we cannot afford to store too many images in the replay buffer.
            self.device,
            store_image=self.cfg.image_reward,
            image_size=image_height)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        reward_model_class = RewardModel
        if self.reward == 'learn_from_preference':
            reward_model_class = RewardModel
        elif self.reward == 'learn_from_score':
            reward_model_class = RewardModelScore
        if not self.multiple:
            self.reward_model = reward_model_class(
                ### original PEBBLE parameters
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                ensemble_size=cfg.ensemble_size,
                size_segment=cfg.segment,
                activation=cfg.activation, 
                lr=cfg.reward_lr,
                mb_size=cfg.reward_batch, 
                large_batch=cfg.large_batch, 
                label_margin=cfg.label_margin, 
                teacher_beta=cfg.teacher_beta, 
                teacher_gamma=cfg.teacher_gamma, 
                teacher_eps_mistake=cfg.teacher_eps_mistake, 
                teacher_eps_skip=cfg.teacher_eps_skip, 
                teacher_eps_equal=cfg.teacher_eps_equal,
                capacity=cfg.max_feedback * 2,
                
                ### vlm parameters
                vlm_label=cfg.vlm_label,
                vlm=cfg.vlm,
                env_name=cfg.env,
                clip_prompt=clip_env_prompts[cfg.env],
                log_dir=self.logger._log_dir,
                flip_vlm_label=cfg.flip_vlm_label,
                cached_label_path=cfg.cached_label_path,

                ### image-based reward model parameters
                image_reward=cfg.image_reward,
                image_height=image_height,
                image_width=image_width,
                resize_factor=self.resize_factor,
                resnet=cfg.resnet,
                conv_kernel_sizes=cfg.conv_kernel_sizes,
                conv_strides=cfg.conv_strides,
                conv_n_channels=cfg.conv_n_channels,
            )
        else:
            self.reward_model = []
            for i in range(4):
                rew_model =  reward_model_class(
                ### original PEBBLE parameters
                self.env.observation_space.shape[0],
                self.env.action_space.shape[0],
                ensemble_size=cfg.ensemble_size,
                size_segment=cfg.segment,
                activation=cfg.activation, 
                lr=cfg.reward_lr,
                mb_size=cfg.reward_batch, 
                large_batch=cfg.large_batch, 
                label_margin=cfg.label_margin, 
                teacher_beta=cfg.teacher_beta, 
                teacher_gamma=cfg.teacher_gamma, 
                teacher_eps_mistake=cfg.teacher_eps_mistake, 
                teacher_eps_skip=cfg.teacher_eps_skip, 
                teacher_eps_equal=cfg.teacher_eps_equal,
                capacity=cfg.max_feedback * 2,
                
                ### vlm parameters
                vlm_label=cfg.vlm_label,
                vlm=cfg.vlm,
                env_name=cfg.env,
                clip_prompt=clip_env_prompts[cfg.env],
                log_dir=self.logger._log_dir,
                flip_vlm_label=cfg.flip_vlm_label,
                cached_label_path=cfg.cached_label_path,

                ### image-based reward model parameters
                image_reward=cfg.image_reward,
                image_height=image_height,
                image_width=image_width,
                resize_factor=self.resize_factor,
                resnet=cfg.resnet,
                conv_kernel_sizes=cfg.conv_kernel_sizes,
                conv_strides=cfg.conv_strides,
                conv_n_channels=cfg.conv_n_channels,
                )
                self.reward_model.append(copy.deepcopy(rew_model))
        print(self.cfg.reward_model_load_dir)
        if self.cfg.reward_model_load_dir is not None:
            if not self.multiple:
                print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
                self.reward_model.load(self.cfg.reward_model_load_dir, cfg.reward_model_load_step) 
            else:
                print("loading multiple reward models")
                for i in range(4):
                    curr_path =  self.cfg.reward_model_load_dir + "/" + self.model_paths[i]
                    print("loading reward model {} at {}".format(i, curr_path))
                    self.reward_model[i].load(curr_path, cfg.reward_model_load_step) 
                    
                    
                
        if self.cfg.agent_model_load_dir is not None:
            print("loading agent model at {}".format(self.cfg.agent_model_load_dir))
            self.agent.load(self.cfg.agent_model_load_dir, cfg.agent_load_step) 
        
        # self.collect_data(save_additional=False)
    
    def process_and_relabel_data(self, folder_path):
        """
        Reads pickle files from a folder, relabels data using images, and saves a new pickle with specific keys.

        Args:
            folder_path (str): Path to the folder containing the pickle files.
        """
        # Initialize the data dictionary
        data = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "terminals": [],
            "info": [],
            "rewards_pred": []
        }

        # Get list of all pickle files in the folder
        pickle_files = glob.glob(os.path.join(folder_path, '*.pkl'))
        pickle_files.sort()  # Optional: sort the files if order matters

        print(f"Found {len(pickle_files)} pickle files in '{folder_path}'. Processing...")

        for pickle_file in tqdm(pickle_files):
            # Load data from pickle
            with open(pickle_file, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Append data to the main data dictionary
            data["observations"].extend(loaded_data.get("observations", []))
            data["actions"].extend(loaded_data.get("actions", []))
            data["next_observations"].extend(loaded_data.get("next_observations", []))
            data["rewards"].extend(loaded_data.get("rewards", []))
            data["terminals"].extend(loaded_data.get("terminals", []))
            data["info"].extend(loaded_data.get("info", []))
            
            # Retrieve images for relabeling
            next_images = loaded_data.get("images", [])
            if not next_images:
                next_images = loaded_data.get("next_images", [])
                if not next_images:
                    print(f"No 'next_images' found in {pickle_file}. Skipping relabeling for this file.")
                    continue

            # Perform relabeling using images
            rewards_pred = self.relabel_images(next_images)
            data["rewards_pred"].extend(rewards_pred)
            print(f"Relabeled {len(rewards_pred)} rewards from {pickle_file}")

        # Convert lists to numpy arrays
        for key in data.keys():
            data[key] = np.array(data[key])

        # Save the final data to a pickle
        output_pickle_path = os.path.join(self.logger._log_dir, 'processed_data.pkl')
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nProcessed data saved at {output_pickle_path}")
        print(f"Size of the dataset: {len(data['observations'])}")
    
    def process_and_relabel_data_multiple(self, folder_path):
        """
        Reads pickle files from a folder, relabels data using images, and saves a new pickle with specific keys.

        Args:
            folder_path (str): Path to the folder containing the pickle files.
        """
        # Initialize the data dictionary
        data = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "terminals": [],
            "info": [],
            "rewards_0-25": [],
            "rewards_0-50": [],
            "rewards_0-75":[],
            "rewards_prox_flip":[]
        }

        # Get list of all pickle files in the folder
        pickle_files = glob.glob(os.path.join(folder_path, '*.pkl'))
        pickle_files.sort()  # Optional: sort the files if order matters

        print(f"Found {len(pickle_files)} pickle files in '{folder_path}'. Processing...")

        for pickle_file in tqdm(pickle_files):
            # Load data from pickle
            with open(pickle_file, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Append data to the main data dictionary
            data["observations"].extend(loaded_data.get("observations", []))
            data["actions"].extend(loaded_data.get("actions", []))
            data["next_observations"].extend(loaded_data.get("next_observations", []))
            data["rewards"].extend(loaded_data.get("rewards", []))
            data["terminals"].extend(loaded_data.get("terminals", []))
            data["info"].extend(loaded_data.get("info", []))
            
            # Retrieve images for relabeling
            next_images = loaded_data.get("images", [])
            if not next_images:
                next_images = loaded_data.get("next_images", [])
                if not next_images:
                    print(f"No 'next_images' found in {pickle_file}. Skipping relabeling for this file.")
                    continue

            # Perform relabeling using images
            rewards_pred = self.relabel_images_multiple(next_images)
            data["rewards_0-25"].extend(rewards_pred[0])
            data["rewards_0-50"].extend(rewards_pred[1])
            data["rewards_0-75"].extend(rewards_pred[2])
            data["rewards_prox_flip"].extend(rewards_pred[3])
            print(f"Relabeled {len(rewards_pred[0])} rewards from {pickle_file}")

        # Convert lists to numpy arrays
        for key in data.keys():
            data[key] = np.array(data[key])

        # Save the final data to a pickle
        output_pickle_path = os.path.join(self.logger._log_dir, 'processed_data.pkl')
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nProcessed data saved at {output_pickle_path}")
        print(f"Size of the dataset: {len(data['observations'])}")
    
    
    def process_and_relabel_data_clip(self, folder_path):
        """
        Reads pickle files from a folder, relabels data using images, and saves a new pickle with specific keys.
        
        In addition to relabeling rewards via self.relabel_images, this function also computes a CLIP-based
        predicted reward using clip_image_text_matching, and stores it as "clip-score".
        
        Args:
            folder_path (str): Path to the folder containing the pickle files.
        """
        # Ensure clip_env_prompts and clip_image_text_matching are imported in your module

        # Initialize the data dictionary with an extra key for CLIP scores
        data = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "terminals": [],
            "info": [],
            # "rewards_pred": [],
            "clip-score": []  # new key for CLIP-based predicted rewards
        }

        # Get list of all pickle files in the folder
        pickle_files = glob.glob(os.path.join(folder_path, '*.pkl'))
        pickle_files.sort()  # Optional: sort the files if order matters

        print(f"Found {len(pickle_files)} pickle files in '{folder_path}'. Processing...")

        for pickle_file in tqdm(pickle_files):
            # Load data from pickle
            with open(pickle_file, 'rb') as f:
                loaded_data = pickle.load(f)
                print(loaded_data.keys())
            
            # Append data to the main data dictionary
            data["observations"].extend(loaded_data.get("observations", []))
            data["actions"].extend(loaded_data.get("actions", []))
            data["next_observations"].extend(loaded_data.get("next_observations", []))
            data["rewards"].extend(loaded_data.get("rewards", []))
            data["terminals"].extend(loaded_data.get("terminals", []))
            data["info"].extend(loaded_data.get("info", []))
            
            # Retrieve images for relabeling
            next_images = loaded_data.get("images", [])
            if not next_images:
                next_images = loaded_data.get("next_images", [])
                if not next_images:
                    print(f"No 'next_images' found in {pickle_file}. Skipping relabeling for this file.")
                    continue
            
            # Compute CLIP-based predicted rewards using clip_image_text_matching
            # Get the prompt for the current environment
            query_prompt = clip_env_prompts[self.cfg.env]
            print("Querying clip-scores")
            clip_score = self.clip_infer_scores(next_images, query_prompt)
            data["clip-score"].extend(clip_score)
            print(f"Computed {len(clip_score)} clip scores from {pickle_file}")
            # Perform relabeling using images (for rewards_pred)
            # rewards_pred = self.relabel_images(next_images)
            # data["rewards_pred"].extend(rewards_pred)
            # print(f"Relabeled {len(rewards_pred)} rewards from {pickle_file}")
            del loaded_data
            

        # Convert lists to numpy arrays
        for key in data.keys():
            print(key)
            data[key] = np.array(data[key])

        # Save the final data to a pickle
        output_pickle_path = os.path.join(self.logger._log_dir, 'processed_data.pkl')
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nProcessed data saved at {output_pickle_path}")
        print(f"Size of the dataset: {len(data['observations'])}")

    def collect_data(self, save_additional=False, collect_images=True, save_interval=250):
        print("Epsilon: ", self.cfg.epsilon)
        data = {}
        data["observations"] = []
        data["actions"] = []
        data["next_observations"] = []
        data["rewards"] = []
        data["terminals"] = []
        data["info"] = []
        if collect_images:
            data["images"] = []
            # data["next images"] = []
        else:
            data["rewards_pred"] = []
        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        all_ep_infos = []
        for episode in tqdm(range(self.cfg.num_eval_episodes + 10)):
            state, images, actions, next_state, next_images, rewards, episode_return, terminals, info = self.collect_episode(episode, save_additional=save_additional)
            data["observations"] += state
            data["actions"] += actions
            data["next_observations"] += next_state
            data["rewards"] += rewards
            data["terminals"] += terminals
            data["info"] += info
            if collect_images:
                data["images"] += images
                # data["next images"] += next_images
            else:
                rewards_pred = self.relabel_images(next_images)
                data["rewards_pred"] += rewards_pred

            if episode%save_interval == 0 and episode > 0:
                data["observations"] = np.array(data["observations"])
                data["actions"] = np.array(data["actions"])
                data["next_observations"] = np.array(data["next_observations"])
                data["rewards"] = np.array(data["rewards"])
                data["terminals"] = np.array(data["terminals"])
                # data["info"] = np.array(data["info"])
                
                ### save the collected demos
                if collect_images:
                    # data["images"] = np.array(data["images"])
                    # data["next images"] = np.array(data["next images"])
                    self.relabel(data)
                else:
                    data["rewards_pred"] = np.array(data["rewards_pred"])
                
                with open(f"{self.logger._log_dir}/data_{episode/save_interval}.pkl", "wb") as f:
                    pickle.dump(data, f)
                print("saved data: ", episode/save_interval)
                data["observations"] = []
                data["actions"] = []
                data["next_observations"] = []
                data["rewards"] = []
                data["terminals"] = []
                data["info"] = []
                if collect_images:
                    data["images"] = []
                    # data["next images"] = []
                else:
                    data["rewards_pred"] = []

        print("Completed data collection")
        print("Data saved at {}".format(self.logger._log_dir))
        print("Size of the dataset: ", len(data["observations"]))
    
    def relabel(self, data):
        if not self.cfg.image_reward:
            batch_size = 200
        else:
            batch_size = 32
        self.idx = len(data['observations'])
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
        if  "rewards_pred" not in data:
            data["rewards_pred"] = np.empty_like(data["rewards"])
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
            
            if not self.cfg.image_reward:
                obses = data["observations"][index*batch_size:last_index]
                actions = data["actions"][index*batch_size:last_index]
                inputs = np.concatenate([obses, actions], axis=-1)
            else:
                inputs = copy.deepcopy(data["images"][index*batch_size:last_index])
                inputs = np.array(inputs)
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                inputs = inputs.astype(np.float32) / 255.0

            pred_reward = self.reward_model.r_hat_batch(inputs)
            data["rewards_pred"][index*batch_size:last_index] = np.squeeze(pred_reward)
            del inputs
        torch.cuda.empty_cache()      
        
    def relabel_images(self, images):
        pred = []
        next_images = np.array(images)
        idx = len(next_images)
        batch_size = 32
        total_iter = int(idx/batch_size)
        if idx > batch_size*total_iter:
            total_iter += 1
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > idx:
                last_index = idx
            inputs = next_images[index*batch_size:last_index]
            inputs = np.transpose(inputs, (0, 3, 1, 2))
            inputs = inputs.astype(np.float32) / 255.0
            pred_reward = self.reward_model.r_hat_batch(inputs)
            pred += list(np.squeeze(pred_reward))
        
        del next_images
        del inputs
        torch.cuda.empty_cache()
        return pred
    
    def relabel_images_multiple(self, images):
        pred = [[],[],[],[]]
        next_images = np.array(images)
        idx = len(next_images)
        batch_size = 32
        total_iter = int(idx/batch_size)
        if idx > batch_size*total_iter:
            total_iter += 1
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > idx:
                last_index = idx
            inputs = next_images[index*batch_size:last_index]
            inputs = np.transpose(inputs, (0, 3, 1, 2))
            inputs = inputs.astype(np.float32) / 255.0
            for i in range(4):
                pred_reward = self.reward_model[i].r_hat_batch(inputs)
                pred[i] += list(np.squeeze(pred_reward))
        
        del next_images
        del inputs
        torch.cuda.empty_cache()
        return pred
        

    def collect_episode(self, episode, save_additional=False, save_vid=False):
        # print("evaluating episode {}".format(episode))
        images = []
        next_images = []
        actions = []
        state = []
        next_state = []
        rewards = []
        terminals = []
        info = []
        epsilon = self.cfg.epsilon
        obs = self.env.reset()
        if "metaworld" in self.cfg.env:
            obs = obs[0]

        self.agent.reset()
        done = False
        episode_reward = 0
        true_episode_reward = 0
        if self.log_success:
            episode_success = 0
        t_idx = 0
        env_name = self.cfg.env
        image_height = self.image_height
        image_width = self.image_width
        
        if "metaworld" in env_name:
            rgb_image = self.env.render()
            rgb_image = rgb_image[::-1, :, :]
            if "drawer" in env_name or "sweep" in env_name:
                rgb_image = rgb_image[100:400, 100:400, :]
        elif env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            rgb_image = self.env.render(mode='rgb_array')
        elif 'softgym' in env_name:
            rgb_image = self.env.render(mode='rgb_array', hide_picker=True)
        else:
            rgb_image = self.env.render(mode='rgb_array')
        
        image = cv2.resize(rgb_image, (image_height, image_width)) # NOTE: resize image here
    
            
        while not done:
            state += [obs]
            images += [image]
            with utils.eval_mode(self.agent):
                rand =  np.random.uniform(low=0.0, high=1.0)
                if rand<epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(obs, sample=False)
            try:
                obs, reward, done, extra = self.env.step(action)
            except:
                obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated

            if "metaworld" in env_name:
                rgb_image = self.env.render()
                rgb_image = rgb_image[::-1, :, :]
                if "drawer" in env_name or "sweep" in env_name:
                    rgb_image = rgb_image[100:400, 100:400, :]
            elif env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                rgb_image = self.env.render(mode='rgb_array')
            elif 'softgym' in env_name:
                rgb_image = self.env.render(mode='rgb_array', hide_picker=True)
            else:
                rgb_image = self.env.render(mode='rgb_array')
            
            image = cv2.resize(rgb_image, (image_height, image_width)) # NOTE: resize image here
           
            episode_reward += reward
            true_episode_reward += reward
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
            actions += [action]
            rewards +=[reward]
            next_state +=[obs]
            next_images += [image]
            terminals +=[done]
            info += [extra]
            t_idx += 1
            if self.cfg.mode == 'eval' and t_idx > 200:
                break
                
        if 'softgym' in self.cfg.env:
            images = self.env.video_frames[:-1]
            next_images = self.env.video_frames[1:]
            
        if save_vid:
            save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
            if not os.path.exists(save_gif_dir):
                os.makedirs(save_gif_dir)
        
            save_gif_path = os.path.join(save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(self.step, episode, round(true_episode_reward, 2)))
            utils.save_numpy_as_gif(np.array(images), save_gif_path)
            
        if save_additional:
            save_image_dir = os.path.join(self.logger._log_dir, 'eval_images')
            if not os.path.exists(save_image_dir):
                os.makedirs(save_image_dir)
            for i, image in enumerate(images):
                save_image_path = os.path.join(save_image_dir, 'step{:07}_episode{:02}_{}.png'.format(self.step, episode, i))
                image = Image.fromarray(image)
                image.save(save_image_path)
            save_reward_path = os.path.join(self.logger._log_dir, "eval_reward")
            if not os.path.exists(save_reward_path):
                os.makedirs(save_reward_path)
            with open(os.path.join(save_reward_path, "step{:07}_episode{:02}.pkl".format(self.step, episode)), "wb") as f:
                pkl.dump(rewards, f)
        # print("Episode length: ", len(images))
        return state, images, actions, next_state, next_images, rewards, episode_reward, terminals, info
    
    def clip_infer_scores(self, images, text, batch_size=32):
        """
        Compute CLIP similarity scores for a list of images given a text prompt.
        
        This function:
        - Converts the list of images to a numpy array.
        - Precomputes the text embedding only once.
        - Processes the images in batches of size `batch_size` (handling any leftover images).
        - Uses the CLIP model to compute similarity scores for each image.
        
        Args:
            images (list or array): List of images (each as a numpy array).
            text (str): The text prompt.
            batch_size (int, optional): Batch size for processing images. Defaults to 64.
            
        Returns:
            list: Similarity scores (floats) for each image.
        """
        scores = []
        next_images = np.array(images)
        idx = len(next_images)
        print("Prompt: ", text)
        print("size: ", idx)
        # Precompute the text embedding once.
        tokenized_text = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Determine the number of iterations (batches)
        total_iter = int(idx / batch_size)
        if idx > batch_size * total_iter:
            total_iter += 1
            
        for index in tqdm(range(total_iter)):
            last_index = (index + 1) * batch_size
            if last_index > idx:
                last_index = idx
            batch_imgs = next_images[index * batch_size : last_index]
            
            # Process each image: convert to PIL, ensure RGB, and preprocess.
            processed_images = []
            for img in batch_imgs:
                pil_img = Image.fromarray(img).convert('RGB')
                processed_images.append(preprocess(pil_img))
            
            image_tensor = torch.stack(processed_images).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # Compute similarity scores (dot product) for the batch.
                batch_similarity = (image_features @ text_features.T).squeeze(1)
                batch_similarity = batch_similarity**2 - 1
            scores += list(batch_similarity.cpu().numpy())
        del next_images, processed_images, image_tensor
        torch.cuda.empty_cache()
        return scores

    
@hydra.main(config_path='/project_data/held/sreyas/RL-VLM-F/config/datagen_softgym.yaml', strict=True)
def main(cfg):
    print(cfg)
    print("Loading agent step: ", cfg.agent_load_step)
    print("Loading reward model step: ", cfg.reward_model_load_step)
    print("Epsilon: ", cfg.epsilon)
    workspace = DataGen(cfg)
    workspace.process_and_relabel_data_multiple(cfg.dataset_dir)
    print("Save interval :", cfg.num_eval_episodes)
    print("Data collection completed , bye bye")
   

if __name__ == "__main__":
    main()      
    
    
        
        