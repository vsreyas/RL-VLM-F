import sys
import os

sys.path.append("/home/sreyas/Desktop/RL-VLM-F/agent")
from flowdiffusion.flowdiffusion.goal_diffusion_policy import GoalGaussianDiffusion as GoalGaussianDiffusionPolicy, Trainer as TrainerPolicy
from flowdiffusion.flowdiffusion.diffusion_policy_baseline.unet import Unet1D
from flowdiffusion.flowdiffusion.diffusion_policy_baseline.dataset import MWDataset
from torchvision import transforms as T
from einops import rearrange
import torch
from PIL import Image
from torch import nn
import numpy as np
import cv2
import pyrallis
import wandb
import utils
from dataclasses import asdict, dataclass
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym
import random


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "CartPole-v1"  # OpenAI gym environment name
    seed: int = 42  # Sets Gym, PyTorch and Numpy seeds
    eval_iter :int = 25 #Number of evaluations when running eval method - default 10
    eval_freq: int = int(2000)  # How often (time steps) we evaluate -default 5000
    n_episodes: int = 5  # How many episodes run during evaluation
    max_timesteps: int = int(30000)  # Max time steps to run environment - defualt int (1e6)
    dataset_dir: str = "/mnt/sda1/sreyas/RL_VLM_F-exp/datagen/Cartpole/datagen_Cartpole-Expert"  # Where to load dataset
    results_folder: str = "/home/sreyas/Desktop/RL-VLM-F/diffusion/cartpole/"  # Where to save results
    milestone: Optional[int] = None   # Model load file name, "" doesn't load
    render: bool = True #render and save outputs in eval
    horizon_length: int = 8
    # IQL
  
    # Wandb logging
    project: str = "RISS"
    group: str = "Diff-Policy"
    name: str = "Diff-Policy"
    
def set_seed(
    seed: int, env = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    
def get_diffusion_policy(ckpt_dir='/home/venky/Desktop/RL-VLM-F/agent/results/ckpts/diffusion_policy', milestone=None, sampling_timesteps=10, dataset = None, obs_dim=39, action_dim=4, env = None, horizon_length=4, config=None):
    print("Action dim: ", action_dim)
    print("Obs dim: ", obs_dim)
    unet = Unet1D(action_space=action_dim, obs_steps=2, obs_dim=obs_dim)

    diffusion = GoalGaussianDiffusionPolicy(
        channels=action_dim,
        model=unet,
        image_size=horizon_length,
        timesteps=100,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    if dataset is None:
        dataset = [0]
    trainer = TrainerPolicy(
        diffusion_model=diffusion,
        train_set=dataset,
        valid_set=[0],
        train_lr=1e-4,
        train_num_steps =config.max_timesteps,
        save_and_sample_every =config.eval_freq,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =32,
        valid_batch_size =1,
        gradient_accumulate_every = 1,
        num_samples=1, 
        results_folder =ckpt_dir,
        fp16 =True,
        amp=True,
    )
    if milestone is not None:
        print("Loading checkpoint from milestone: ", milestone)
        trainer.load(milestone)
    return trainer

class DiffusionPolicy:
    def __init__(policy, env, milestone=None, amp=True, sampling_timesteps=10, dataset=None, results_folder="/results", config=None):
        policy.env = env  # Store the environment
        policy.policy = get_diffusion_policy(ckpt_dir=results_folder, env=env, milestone=milestone, sampling_timesteps=sampling_timesteps, dataset=dataset, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], horizon_length=config.horizon_length, config=config)
        policy.amp = amp
        policy.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])
        policy.policy.env = env
        policy.policy.results_folder = Path(config.results_folder)
        policy.policy.seed = config.seed 
        policy.policy.n_episodes = config.n_episodes
        policy.policy.eval_iter = config.eval_iter
        policy.policy.save_and_sample_every = config.eval_freq
        policy.policy.train_num_steps = config.max_timesteps
        policy.policy.env_name = config.env
        policy.policy.config = config

def wandb_init(config: dict) -> None:
    wandb.init(
        # mode="disabled",
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
    
@pyrallis.wrap()
def run_diff_policy(config: TrainConfig):
    
    config.name = config.name + "-seed-" + str(config.seed)
    
    config.name = f"{config.name}-{config.env}-{str(uuid.uuid4())[:8]}"
    if config.results_folder is not None:
        config.results_folder = os.path.join(config.results_folder, config.name)
    config.gif_path = os.path.join(config.results_folder, "eval_gifs")
    os.makedirs(config.results_folder, exist_ok=True)
    os.makedirs(config.gif_path, exist_ok=True)
    
    if 'metaworld' in config.env:
        env = utils.make_metaworld_env(config)
    elif config.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
        env = utils.make_classic_control_env(config)
    elif 'softgym' in config.env:
        env = utils.make_softgym_env(config)
    else:
        env = utils.make_env(config)
        
    set_seed(config.seed, env=env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    dataset = MWDataset(dataset_dir=config.dataset_dir, output_dir=config.results_folder, horizon_length=config.horizon_length)
    policy = DiffusionPolicy(env=env, dataset=dataset,config=config, results_folder=config.results_folder, milestone=config.milestone)# Pass environment to policy
    
    with open(os.path.join(config.results_folder, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    
    wandb_init(asdict(config))
    policy.policy.train()
    

@pyrallis.wrap()
def eval_diff_policy(config: TrainConfig):
    
    # config.name = config.name + "-seed-" + str(config.seed)
    
    # config.name = f"{config.name}-{config.env}-{str(uuid.uuid4())[:8]}"
    # if config.results_folder is not None:
    #     config.results_folder = os.path.join(config.results_folder, config.name)
    # config.gif_path = os.path.join(config.results_folder, "eval_gifs")
    # os.makedirs(config.results_folder, exist_ok=True)
    # os.makedirs(config.gif_path, exist_ok=True)
    device ="cuda" if torch.cuda.is_available() else "cpu"
    if 'metaworld' in config.env:
        env = utils.make_metaworld_env(config)
    elif config.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
        env = utils.make_classic_control_env(config)
    elif 'softgym' in config.env:
        env = utils.make_softgym_env(config)
    else:
        env = utils.make_env(config)
        
    set_seed(config.seed, env=env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    dataset = MWDataset(dataset_dir=config.dataset_dir, output_dir=config.results_folder, horizon_length=config.horizon_length)
    policy = DiffusionPolicy(env=env, dataset=dataset,config=config, results_folder=config.results_folder, milestone=config.milestone)# Pass environment to policy
    
    with open(os.path.join(config.results_folder, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    
    wandb_init(asdict(config))
    if "metaworld" in policy.policy.config.env:
        eval_scores, success, mean_obj_to_target = policy.policy.eval_actor(
            device=device,
            n_episodes=config.n_episodes,
            env_name=config.env,
            step=10000
        )
    else:
        eval_scores, success = policy.policy.eval_actor(
            device=device,
            n_episodes=config.n_episodes,
            env_name=config.env,
            step=10000
        )
    eval_score = eval_scores.mean()
    #normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
    evaluations.append(eval_score)
    print("---------------------------------------")
    print(
        f"Evaluation over {config.n_episodes} episodes: "
        f"{eval_score:.3f} "
    )
    print(
        f"Success percentage over {config.n_episodes} episodes: "
        f"{success:.3f} "
    )
    if "metaworld" in config.env:
        print(
            f"Mean object to target distance over {config.n_episodes} episodes: "
            f"{mean_obj_to_target:.3f} "
        )

    if "metaworld" in config.env:
        wandb.log(
            {"eval_score": eval_score, "success_percent":success, "object_to_target_distance":mean_obj_to_target}
        )
    else:
        wandb.log(
            {"eval_score": eval_score, "success_percent":success}
        )

    
if __name__ == "__main__":
    run_diff_policy()
    # eval_diff_policy()
   
