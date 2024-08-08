#!/usr/bin/env python3
import numpy as np
import torch
import os
import time
import pickle as pkl

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from reward_model_score import RewardModelScore
from collections import deque
from prompt import clip_env_prompts

import utils
import hydra
from PIL import Image

from vlms.blip_infer_2 import blip2_image_text_matching
from vlms.clip_infer import clip_infer_score as clip_image_text_matching
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
        print(f'workspace: {self.work_dir}')

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
        
        if self.cfg.reward_model_load_dir is not  None:
            print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
            self.reward_model.load(self.cfg.reward_model_load_dir, cfg.reward_model_load_step) 
                
        if self.cfg.agent_model_load_dir is not None:
            print("loading agent model at {}".format(self.cfg.agent_model_load_dir))
            self.agent.load(self.cfg.agent_model_load_dir, cfg.agent_load_step) 
        
        self.collect_data(save_additional=False)
        
    def collect_data(self, save_additional=False, collect_images=False, save_interval=1000):
        data = {}
        data["observations"] = []
        data["actions"] = []
        data["next_observations"] = []
        data["rewards"] = []
        data["terminals"] = []
        data["info"] = []
        if collect_images:
            data["images"] = []
            data["next images"] = []
        else:
            data["rewards_pred"] = []
        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        all_ep_infos = []
        for episode in tqdm(range(self.cfg.num_eval_episodes)):
            state, images, actions, next_state, next_images, rewards, episode_return, terminals, info = self.collect_episode(episode, save_additional=save_additional)
            data["observations"] += state
            data["actions"] += actions
            data["next_observations"] += next_state
            data["rewards"] += rewards
            data["terminals"] += terminals
            data["info"] += info
            if collect_images:
                data["images"] += images
                data["next images"] += next_images
            else:
                rewards_pred = self.relabel_images(next_images)
                data["rewards_pred"] += rewards_pred

            # if episode%save_interval == 0 and episode > 0:
            #     with open(f"{self.logger._log_dir}/data_raw_{episode}.pkl", "wb") as f:
            #         pickle.dump(data, f)
        
        data["observations"] = np.array(data["observations"])
        data["actions"] = np.array(data["actions"])
        data["next_observations"] = np.array(data["next_observations"])
        data["rewards"] = np.array(data["rewards"])
        data["terminals"] = np.array(data["terminals"])
        data["info"] = np.array(data["info"])
        
        ### save the collected demos
        if collect_images:
            data["images"] = np.array(data["images"])
            data["next images"] = np.array(data["next images"])
            self.relabel(data)
        else:
            data["rewards_pred"] = np.array(data["rewards_pred"])
        
        with open(f"{self.logger._log_dir}/data.pkl", "wb") as f:
            pickle.dump(data, f)

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
                inputs = data["next images"][index*batch_size:last_index]
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                inputs = inputs.astype(np.float32) / 255.0

            pred_reward = self.reward_model.r_hat_batch(inputs)
            data["rewards_pred"][index*batch_size:last_index] = np.squeeze(pred_reward)
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
        

    def collect_episode(self, episode, save_additional=True, save_vid=False):
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
        if "metaworld" in self.cfg.env:
            rgb_image = self.env.render()
            rgb_image = rgb_image[::-1, :, :]
            if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                rgb_image = rgb_image[100:400, 100:400, :]
        elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            rgb_image = self.env.render(mode='rgb_array')
        else:
            rgb_image = self.env.render(mode='rgb_array')
            
        while not done:
            state += [obs]
            images += [rgb_image]
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

            if "metaworld" in self.cfg.env:
                rgb_image = self.env.render()
                if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                    rgb_image = rgb_image[100:400, 100:400, :]
            elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                rgb_image = self.env.render(mode='rgb_array')
            else:
                rgb_image = self.env.render(mode='rgb_array')

           
            episode_reward += reward
            true_episode_reward += reward
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
            actions += [action]
            rewards +=[reward]
            next_state +=[obs]
            next_images += [rgb_image]
            terminals +=[done]
            info += [extra]
            t_idx += 1
            if self.cfg.mode == 'eval' and t_idx > 100:
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
        print("Episode length: ", len(images))
        return state, images, actions, next_state, next_images, rewards, episode_reward, terminals, info
    
@hydra.main(config_path='/home/sreyas/RL-VLM-F/RL-VLM-F/config/datagen_softgym.yaml', strict=True)
def main(cfg):
    print("Loading agent step: ", cfg.agent_load_step)
    print("Loading reward model step: ", cfg.reward_model_load_step)
    print("Epsilon: ", cfg.epsilon)
    workspace = DataGen(cfg)
    print("Save interval :", cfg.num_eval_episodes)
    print("Data collection completed , bye bye")
   

if __name__ == "__main__":
    main()      
    
    
        
        