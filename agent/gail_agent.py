import sys
import os
sys.path.append(os.getcwd())
import copy
import random
import uuid
import json
import pickle as pkl
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym
import numpy as np
import pyrallis
import torch
import gym
import utils
import wandb

from agent.gail_pytorch.models.nets import Expert
from agent.gail_pytorch.models.gail import GAIL

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

def make_softgym_env(cfg):
    env_name = cfg.env.replace('softgym_','')
    env_kwargs = env_arg_dict[env_name]
    env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))

    return env

TensorBatch = List[torch.Tensor]

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "metaworld_drawer-open-v2"  # OpenAI gym environment name
    data_set_path: Optional[str] = "/home/venky/vanitch/drawer_open/drawer_open-expert.pkl"
    seed: int = 43  # Sets Gym, PyTorch and Numpy seeds
    eval_iter :int = 10 #Number of evaluations when running eval method
    eval_freq: int = int(500)  # How often (time steps) we evaluate -default 5000
    n_episodes: int = 10  # How many episodes run during evaluation
    checkpoints_path: Optional[str] = "/home/venky/Desktop/RL-VLM-F/offline_rl/drawer_open"  # Save path
    load_model: str =""   # Model load file name, "" doesn't load
    render: bool = True #render and save outputs in eval
    # IQL
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    num_iters: int = 10000
    horizon: Optional[int] = None
    lambda_: float = 0.01
    gae_gamma: float = 0.99
    gae_lambda: float = 0.99
    epsilon: float = 0.05
    max_kl: float = 0.1
    cg_damping: float = 0.1
    normalize_advantage: bool = True
    lr: float = 1e-3
    num_samples: int = 512
    policy_update_freq: int = 5
    # Wandb logging
    project: str = "debug_"
    group: str = "metaworld"
    name: str = "GAIL-"

def render(env, env_name,image_height=200,image_width=200):
    if "metaworld" in env_name:
            rgb_image = env.render()
            rgb_image = rgb_image[::-1, :, :]
            if "drawer" in env_name or "sweep" in env_name:
                rgb_image = rgb_image[100:400, 100:400, :]
    elif env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
        rgb_image = env.render(mode='rgb_array')
    elif 'softgym' in env_name:
        rgb_image = env.render(mode='rgb_array', hide_picker=True)
    else:
        rgb_image = env.render(mode='rgb_array')


    image = cv2.resize(rgb_image, (image_height, image_width)) # NOTE: resize image here
        
    return image
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

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

def make_numpy(data:Dict, images = False):

    data["observations"] = np.array(data["observations"])
    data["actions"] = np.array(data["actions"])
    data["next_observations"] = np.array(data["next_observations"])
    data["rewards"] = np.array(data["rewards"])
    data["terminals"] = np.array(data["terminals"])
    if images:
        data["images"] = np.array(data["images"])
        data["next images"] = np.array(data["next images"])


    return data

def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
        
def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        if len(state) == 2:
            state, _ = state
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

@pyrallis.wrap()
def main(config: TrainConfig):
    config.name = config.name + "-seed-" + str(config.seed)
    
    config.name = f"{config.name}-{config.env}-{str(uuid.uuid4())[:8]}"
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name)
        config.gif_path = os.path.join(config.checkpoints_path, "eval_gifs")
    
    if 'metaworld' in config.env:
        env = utils.make_metaworld_env(config)
        log_success = True
    elif config.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
        env = utils.make_classic_control_env(config)
    elif 'softgym' in config.env:
        env = make_softgym_env(config)
    else:
        env = utils.make_env(config)

    print(config.data_set_path)
    print(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    discrete = False


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    with open(config.data_set_path, 'rb') as f:
            dataset = pkl.load(f)
    dataset = make_numpy(dataset)
    print(dataset.keys())
    print(dataset["rewards"])
    
    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    exp_obs = dataset["observations"]
    exp_acts = dataset["actions"]
    
    model = GAIL(state_dim, action_dim, discrete, config).to(device)
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        os.makedirs(config.gif_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    wandb_init(asdict(config))
    
    results = model.train(env, config.env, config.seed, config.n_episodes, exp_obs, exp_acts, render_=False)

    print("Gail training done")
    print("results", results)   

    
    


if __name__ == "__main__":
    main()
