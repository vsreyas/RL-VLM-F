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

class Offline_Workspace(object):
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
        with open(cfg.dataset_path, 'rb') as f:
                self.dataset = pkl.load(f)
        
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
        
        if self.cfg.reward_model_load_dir != "None":
            print("loading reward model at {}".format(self.cfg.reward_model_load_dir))
            self.reward_model.load(self.cfg.reward_model_load_dir, 1000000) 
                
        if self.cfg.agent_model_load_dir != "None":
            print("loading agent model at {}".format(self.cfg.agent_model_load_dir))
            self.agent.load(self.cfg.agent_model_load_dir, 1000000) 
        
        self.load_dataset_to_buffer()
        
    def evaluate(self, save_additional=False):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        all_ep_infos = []
        for episode in range(self.cfg.num_eval_episodes):
            print("evaluating episode {}".format(episode))
            images = []
            obs = self.env.reset()
            if "metaworld" in self.cfg.env:
                obs = obs[0]

            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            ep_info = []
            rewards = []
            t_idx = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                try:
                    obs, reward, done, extra = self.env.step(action)
                except:
                    obs, reward, terminated, truncated, extra = self.env.step(action)
                    done = terminated or truncated
                ep_info.append(extra)

                rewards.append(reward)
                if "metaworld" in self.cfg.env:
                    rgb_image = self.env.render()
                    if self.cfg.mode != 'eval':
                        rgb_image = rgb_image[::-1, :, :]
                        if "drawer" in self.cfg.env or "sweep" in self.cfg.env:
                            rgb_image = rgb_image[100:400, 100:400, :]
                    else:
                        rgb_image = rgb_image[::-1, :, :]
                elif self.cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
                    rgb_image = self.env.render(mode='rgb_array')
                else:
                    rgb_image = self.env.render(mode='rgb_array')

                if 'softgym' not in self.cfg.env:
                    images.append(rgb_image)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                    
                t_idx += 1
                if self.cfg.mode == 'eval' and t_idx > 50:
                    break
                    
            all_ep_infos.append(ep_info)
            if 'softgym' in self.cfg.env:
                images = self.env.video_frames
                
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
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        for key, value in extra.items():
            self.logger.log('eval/' + key, value, self.step)

        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
            
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
        # get feedbacks
        labeled_queries = 0 
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        total_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    self.reward_model.train()
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    self.reward_model.train()
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break
        
        if self.reward == 'learn_from_preference':
            print("Reward function is updated!! ACC: " + str(total_acc))
        elif self.reward == 'learn_from_score':
            print("Reward function is updated!! MSE: " + str(total_acc))
        return total_acc, self.reward_model.vlm_label_acc

    def run(self):
        model_save_dir = os.path.join(self.work_dir, "models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        

        interact_count = 0
        reward_learning_acc = 0
        vlm_acc = 0
        eval_cnt = 0
        while self.step < self.cfg.num_train_steps:
            
            # update reward function
            if self.total_feedback < self.cfg.max_feedback and (
                self.reward == 'learn_from_preference' or self.reward == 'learn_from_score'):
                if interact_count == self.cfg.num_interact:
                    # update schedule
                    if self.cfg.reward_schedule == 1:
                        frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                        if frac == 0:
                            frac = 0.01
                    elif self.cfg.reward_schedule == 2:
                        frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                    else:
                        frac = 1
                    self.reward_model.change_batch(frac)
                    
                    # corner case: new total feed > max feed
                    if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                        self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                        
                    reward_learning_acc, vlm_acc = self.learn_reward()
                    self.reward_model.eval()
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                    self.reward_model.train()
                    
            self.agent.update(self.replay_buffer, self.logger, self.step, 1)
            self.logger.log('train/reward_learning_acc', reward_learning_acc,
                        self.step)
            self.logger.log('train/vlm_acc', vlm_acc,self.step)
            

            if self.step % self.cfg.save_interval == 0 and self.step > 0:
                self.agent.save(model_save_dir, self.step)
                self.reward_model.save(model_save_dir, self.step)
            
        self.agent.save(model_save_dir, self.step)
        self.reward_model.save(model_save_dir, self.step)
    
    def load_dataset_to_buffer(self):
        size = len(self.dataset["observations"])
        
        if size > self.cfg.replay_buffer_capacity:
            idx = np.random.choice(size, self.cfg.replay_buffer_capacity, replace=False)
        else:
            idx = np.arange(size)
        
        for i in idx:
            obs = self.dataset["observations"][i]
            action = self.dataset["actions"][i]
            next_obs = self.dataset["next_observations"][i]
            reward = self.dataset["rewards"][i]
            done = self.dataset["terminals"][i]
            done = float(done)
            if self.reward == 'blip2_image_text_matching':
                query_image = rgb_image
                query_prompt = clip_env_prompts[self.cfg.env] 
                reward_hat = blip2_image_text_matching(query_image, query_prompt) * 2 - 1 # actually we should scale it [-1, 1] since tanh is used in the reward model
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat
            elif self.reward == 'clip_image_text_matching':
                query_image = rgb_image
                query_prompt = clip_env_prompts[self.cfg.env] 
                reward_hat = clip_image_text_matching(query_image, query_prompt) * 2 - 1 # actually we should scale it [-1, 1] since tanh is used in the reward model
                if self.cfg.flip_vlm_label:
                    reward_hat = -reward_hat
            elif self.reward == 'gt_task_reward':
                reward_hat = reward
            else:
                reward_hat = reward
            if self.cfg.image_reward and self.reward not in ["gt_task_reward", "sparse_task_reward"]:
                rgb_image = self.dataset["images"][i]
                self.replay_buffer.add(obs, action, reward_hat, 
                    next_obs, done, done, image=rgb_image[::self.resize_factor, ::self.resize_factor, :])
            else:
                self.replay_buffer.add(obs, action, reward_hat, 
                    next_obs, done, done)
                
        print("Dataset loaded to buffer!!")
        
@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    print("Save interval :", cfg.save_interval)
    if cfg.mode == 'eval':
        workspace.evaluate(save_additional=cfg.save_images)
        exit()
    workspace.run()

if __name__ == '__main__':
    main()