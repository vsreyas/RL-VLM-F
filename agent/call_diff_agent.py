import sys
import os

sys.path.append("/home/venky/Desktop/RL-VLM-F/agent")
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


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "metaworld_drawer-open-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_iter :int = 25 #Number of evaluations when running eval method - default 10
    eval_freq: int = int(5)  # How often (time steps) we evaluate -default 5000
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment - defualt int (1e6)
    dataset_dir: str = "/home/venky/Desktop/RL-VLM-F/datasets/drawer-open-expert-10_random/42"  # Where to load dataset
    results_folder: str = "/home/venky/Desktop/RL-VLM-F/agent/results"  # Where to save results
    milestone: int = 7   # Model load file name, "" doesn't load
    render: bool = True #render and save outputs in eval
    # IQL
  
    # Wandb logging
    project: str = "RISS"
    group: str = "Diff-Policy"
    name: str = "Diff-Policy"
    
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
    
def get_diffusion_policy(ckpt_dir='/home/venky/Desktop/RL-VLM-F/agent/results/ckpts/diffusion_policy', milestone=7, sampling_timesteps=10, dataset = None, obs_dim=39, action_dim=4, env = None, horizon_length=4):
    unet = Unet1D(action_space=action_dim, obs_steps=2, obs_dim=obs_dim)

    diffusion = GoalGaussianDiffusionPolicy(
        channels=4,
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
        train_num_steps =40000,
        save_and_sample_every =1500,
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
    print("Loading checkpoint from milestone: ", milestone)
    trainer.load(milestone)
    return trainer

class DiffusionPolicy:
    def __init__(self, env, milestone=7, amp=True, sampling_timesteps=10, dataset=None, results_folder="/results", config=None):
        self.env = env  # Store the environment
        self.policy = get_diffusion_policy(env=env, milestone=milestone, sampling_timesteps=sampling_timesteps, dataset=dataset)
        self.amp = amp
        self.transform = T.Compose([
            T.Resize((320, 240)),
            T.CenterCrop((128, 128)),
            T.ToTensor(),
        ])
        self.policy.env = env
        self.policy.results_folder = Path(config.results_folder)
        self.policy.seed = config.seed 
        self.policy.n_episodes = config.n_episodes
        self.policy.eval_iter = config.eval_iter
        self.policy.save_and_sample_every = config.eval_freq
        self.policy.train_num_steps = config.max_timesteps
        self.policy.env_name = config.env
        self.policy.config = config

def wandb_init(config: dict) -> None:
    wandb.init(
        mode="disabled",
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
    if config.checkpoints_path is not None:
        config.checkpoints_path = os.path.join(config.checkpoints_path, config.name)
    config.gif_path = os.path.join(config.checkpoints_path, "eval_gifs")
    
    if 'metaworld' in config.env:
        env = utils.make_metaworld_env(config)
    elif config.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
        env = utils.make_classic_control_env(config)
    elif 'softgym' in config.env:
        env = utils.make_softgym_env(config)
    else:
        env = utils.make_env(config)
        
    set_seed(config.seed, env=env)
    
    dataset = MWDataset(dataset_dir='/home/venky/Desktop/RL-VLM-F/datasets/drawer-open-expert-10_random/42', output_dir=config.results_folder)
    policy = DiffusionPolicy(env=env, dataset=dataset,config=config)# Pass environment to policy
    
    with open(os.path.join(config.results_folder, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    
    wandb_init(asdict(config))
    policy.policy.train()

    
if __name__ == "__main__":
    run_diff_policy()
   
