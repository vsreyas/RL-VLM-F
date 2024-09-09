from .conditional_unet1d import ConditionalUnet1D as Unet1D_diffusion
from .perceiver import PerceiverResampler
from .vis_encoder import ResNet18Encoder
from torch import nn
from einops import rearrange, repeat
import torch

class Unet1D(nn.Module):
    def __init__(self, action_space=4, obs_steps=2, obs_dim=39):
        super(Unet1D, self).__init__()

        self.perceiver = PerceiverResampler(
            dim=512, depth=2
        )
        self.unet = Unet1D_diffusion(
            input_dim=action_space,
            local_cond_dim=None,
            global_cond_dim=512*obs_steps+obs_dim*obs_steps,
            diffusion_step_embed_dim=256,
            down_dims=[256,512,1024],
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False
        )
        self.resnet = ResNet18Encoder()

        self.last_obs = None

    def encode_obs_features(self, obs):
        # obs.shape = (b, f, c, h, w)
        f = obs.shape[1]
        obs = rearrange(obs, 'b f c h w -> (b f) c h w')
        obs = rearrange(self.resnet(obs), '(b f) c -> b (f c)', f=f)
        self.obs_features = obs

    def forward(self, x, t):
        action, obs, obs_ = x
        
        # task_embed = self.perceiver(task_embed).mean(dim=1)
        # obs.shape = (b, f, c, h, w)
        # if self.last_obs is None or not torch.allclose(self.last_obs, obs):
        self.encode_obs_features(obs)
        self.last_obs = obs
        # print("obs_features: ", self.obs_features.shape)
        # print("obs_: ", obs_.shape)
        obs_ = obs_.reshape(obs_.shape[0],-1)
        global_cond = torch.cat([self.obs_features, obs_], dim=1)
        # print("global_cond: ", global_cond.shape)
        # print("action: ", action.shape)
        # print("t: ", t)
        return self.unet(action, t, global_cond=global_cond)

 