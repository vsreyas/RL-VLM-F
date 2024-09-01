import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import Adam

from torchvision import transforms as T, utils
import utils as utils_c
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
import cv2
from accelerate import Accelerator

import matplotlib.pyplot as plt
import numpy as np
import wandb

# from denoising_diffusion_pytorch.version import __version__
__version__ = "0.0"
# from flow_viz import flow_to_image, flow_tensor_to_image
import os

from pynvml import *


def render(env, env_name,image_height=300,image_width=300):
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


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, f, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b f (h c) -> b h c f', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h f d -> b (h d) f', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size=8, #1D
        channels=4,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, obs, clip_x_start=False, rederive_pred_noise=False):
        # task_embed = self.text_encoder(goal).last_hidden_state
        model_output = self.model((x, x_cond, obs), t)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, obs, clip_denoised = False):
        preds = self.model_predictions(x, t, x_cond, obs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, obs):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, obs, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, obs, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            # self_cond = x_start if self.self_condition else None
            img, _ = self.p_sample(img, t, x_cond, obs)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, x_cond, obs, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, obs, clip_x_start = False, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, x_cond, obs, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, image_size, channels), x_cond, obs,  return_all_timesteps = return_all_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, x_cond, obs, noise=None):
        b, f, c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model((x, x_cond, obs), t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, img_cond, obs):
        b, f, c, device, img_size, = *img.shape, img.device, self.image_size
        assert f == img_size, f'height and width of image must be {img_size}, got({f})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, img_cond, obs)

# dataset classes

# class Dataset(Dataset):
#     def __init__(
#         self,
#         folder,
#         image_size,
#         exts = ['jpg', 'jpeg', 'png', 'tiff'],
#         augment_horizontal_flip = False,
#         convert_image_to = None
#     ):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

#         maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

    #     self.transform = T.Compose([
    #         T.Lambda(maybe_convert_fn),
    #         T.Resize(image_size),
    #         T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
    #         T.CenterCrop(image_size),
    #         T.ToTensor()
    #     ])

    # def __len__(self):
    #     return len(self.paths)

    # def __getitem__(self, index):
    #     path = self.paths[index]
    #     img = Image.open(path)
    #     return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        # tokenizer, 
        # text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        cond_drop_chance=0.1,
        seed = 0
    ):
        super().__init__()

        self.cond_drop_chance = cond_drop_chance

        # self.tokenizer = tokenizer
        # self.text_encoder = text_encoder

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model

        self.channels = channels


        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        
        valid_ind = [i for i in range(len(valid_set))][:num_samples]

        train_set = train_set
        valid_set = Subset(valid_set, valid_ind)

        self.ds = train_set
        self.valid_ds = valid_set
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        # dl = dataloader
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = valid_batch_size, shuffle = False, pin_memory = True, num_workers = 4)


        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
            print("EMA initalised")

        # self.text_encoder.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = \
            self.accelerator.prepare(self.model, self.opt)

        # self.text_cache = {}

    @property
    def device(self):
        return self.accelerator.device
        # return "cuda:0"

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__,
            'obs_mean': self.ds.obs_mean,
            'obs_std': self.ds.obs_std,
            'actions_mean': self.ds.actions_mean,
            'actions_std': self.ds.actions_std,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        print(f"loading from {milestone}, path", str(self.results_folder / f'model-{milestone}.pt'))
        
        self.step = data['step']
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        
        opt = self.accelerator.unwrap_model(self.opt)
        self.opt.load_state_dict(data['opt'])
        
        self.model , self.opt = accelerator.prepare(model, self.opt)

        
        # self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
        
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        
        
        self.obs_mean = data['obs_mean']
        self.obs_std = data['obs_std']
        self.actions_mean = data['actions_mean']
        self.actions_std = data['actions_std']

    # @torch.no_grad()
    # def calculate_activation_statistics(self, samples):
    #     assert exists(self.inception_v3)

    #     features = self.inception_v3(samples)[0]
    #     features = rearrange(features, '... 1 1 -> ...')

    #     mu = torch.mean(features, dim = 0).cpu()
    #     sigma = torch.cov(features).cpu()
    #     return mu, sigma

    # def fid_score(self, real_samples, fake_samples):

    #     if self.channels == 1:
    #         real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))

    #     min_batch = min(real_samples.shape[0], fake_samples.shape[0])
    #     real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

    #     m1, s1 = self.calculate_activation_statistics(real_samples)
    #     m2, s2 = self.calculate_activation_statistics(fake_samples)

    #     fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    #     return fid_value
    # def _encode_batch_text(self, batch_text):
    #     batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
    #     batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
    #     return batch_text_embed
    
    # def encode_batch_text(self, batch_text):
    #     out_tokens = []
    #     for i, text in enumerate(batch_text):
    #         if text not in self.text_cache:
    #             self.text_cache[text] = self._encode_batch_text([text])
    #         out_tokens.append(self.text_cache[text])
    #     # pad to max length
    #     max_len = max([t.shape[1] for t in out_tokens])
    #     out_tokens = [F.pad(t, (0, 0, 0, max_len - t.shape[1]), value = 0) for t in out_tokens]
    #     return torch.cat(out_tokens, dim = 0)

    def sample(self, x_conds, obs):
        
        device = self.device
        bs = x_conds.shape[0]
        x_conds = x_conds.to(device)
        obs = obs.to(device)
        # tasks = self.encode_batch_text(tasks).to(device)

        with self.accelerator.autocast():
            output = self.ema.ema_model.sample(batch_size=bs, x_cond=x_conds, obs=obs)
        return output
        

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        print("Training on", device)
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    x = data["action"]
                    x_cond = data["image"]
                    obs = data["observation"]
                    x, x_cond , obs= x.to(device), x_cond.to(device), obs.to(device)

                    # goal_embed = self.encode_batch_text(goal)
                    ### zero whole goal_embed if p < self.cond_drop_chance
                    # goal_embed = goal_embed * (torch.rand(goal_embed.shape[0], 1, 1, device = goal_embed.device) > self.cond_drop_chance).float()


                    with self.accelerator.autocast():
                        loss = self.model(x, x_cond, obs)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                ### get maximum gradient from model
                # max_grad = max([torch.max(torch.abs(p.grad)) for p in self.model.parameters() if p.grad is not None])

                scale = self.accelerator.scaler.get_scale()
                
                pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()
                evaluations = []
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    wandb.log({"loss": total_loss})
                    wandb.log({"loss_scale": scale})
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every
                        # with torch.no_grad():
                        #     milestone = self.step // self.save_and_sample_every
                        #     batches = num_to_groups(self.num_samples, self.valid_batch_size)
                        #     ### get val_imgs from self.valid_dl
                        #     x_conds = []
                        #     xs = []
                        #     obs = []
                        #     # task_embeds = []
                        #     for i, data in enumerate(self.valid_dl):
                        #         xs.append(data["action"].to(device))
                        #         x_conds.append(data['image'].to(device))
                        #         obs.append(data['observation'].to(device))
                        #         # task_embeds.append(self.encode_batch_text(label))
                            
                        #     with self.accelerator.autocast():
                        #         all_xs_list = list(map(lambda n, c, e, o: self.ema.ema_model.sample(batch_size=n, x_cond=c, obs=o), batches, x_conds, obs))
                        
                        # print_gpu_utilization()
                        
                        # gt_xs = torch.cat(xs, dim = 0) # [batch_size, 3*n, 120, 160]
                        # # make it [batchsize*n, 3, 120, 160]
                        # n_rows = gt_xs.shape[1] // 3
                        # gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)
                        # ### save images
                        # x_conds = torch.cat(x_conds, dim = 0).detach().cpu()
                        # obs = torch.cat(obs, dim = 0).detach().cpu()
                        # # x_conds = rearrange(x_conds, 'b (n c) h w -> b n c h w', n=1)
                        # all_xs = torch.cat(all_xs_list, dim = 0).detach().cpu()
                        # all_xs = rearrange(all_xs, 'b (n c) h w -> b n c h w', n=n_rows)

                        # # gt_xs = rearrange(torch.cat([x_conds, gt_xs], dim=1), 'b n c h w -> (b n) c h w', n=n_rows+1)
                        # # pred_xs = rearrange(torch.cat([x_conds, all_xs], dim=1), 'b n c h w -> (b n) c h w', n=n_rows+1)
                        
                        # ### bbox visualization
                        # # label_start, label_end = get_bound_box_labels()
                        # # x_starts = []
                        # # for i, x_start in enumerate(x_conds):
                        # #     x_starts.append(draw_bbox(x_start, label_start[i]))
                        # # x_starts = torch.stack(x_starts).unsqueeze(1)
                        # # x_conds = x_conds.unsqueeze(1)

                        # # x_ends = []
                        # # for i, x_end in enumerate(gt_xs[:, -1]):
                        # #     x_ends.append(draw_bbox(x_end, label_end[i]))
                        # # x_ends = torch.stack(x_ends).unsqueeze(1)



                        # if self.step == self.save_and_sample_every:
                        #     os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
                        #     gt_img = torch.cat([gt_xs[:, :]], dim=1)
                        #     gt_img = rearrange(gt_img, 'b n c h w -> (b n) c h w', n=n_rows)
                        #     utils.save_image(gt_img, str(self.results_folder / f'imgs/gt_img.png'), nrow=n_rows)

                        # os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
                        # pred_img = torch.cat([all_xs[:, :]], dim=1)
                        # pred_img = rearrange(pred_img, 'b n c h w -> (b n) c h w', n=n_rows)
                        # utils.save_image(pred_img, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow=n_rows)

                        #     os.makedirs(str(self.results_folder / f'imgs'), exist_ok = True)
                        #     gt_flow_imgs = [torch.from_numpy(flow_tensor_to_image((gt_flow-0.5)*1000))/255 for gt_flow in gt_flows]
                        #     print(max(gt_flow_imgs[0].numpy().flatten()))
                        #     utils.save_image(gt_flow_imgs, str(self.results_folder / f'imgs/gt_flow.png'), nrow = int(math.sqrt(self.num_samples)))
                        #     utils.save_image(imgs, str(self.results_folder / f'imgs/gt_img.png'), nrow = int(math.sqrt(self.num_samples)))
                        # all_flow_imgs = [torch.from_numpy(flow_tensor_to_image((flow-0.5)*1000))/255 for flow in all_flows]
                        # os.makedirs(str(self.results_folder / f'imgs/outputs'), exist_ok = True)
                        # utils.save_image(all_flow_imgs, str(self.results_folder / f'imgs/outputs/sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)
                        if "drawer" in self.env_name:
                            eval_scores, success, mean_obj_to_target = self.eval_actor(
                                device=self.device,
                                n_episodes=self.config.n_episodes,
                                env_name=self.config.env,
                                step=self.step
                            )
                        else:
                            eval_scores, success = self.eval_actor(
                                device=self.device,
                                n_episodes=self.config.n_episodes,
                                env_name=self.config.env,
                                step=self.step
                            )
                        eval_score = eval_scores.mean()
                        #normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
                        evaluations.append(eval_score)
                        print("---------------------------------------")
                        print(
                            f"Evaluation over {self.config.n_episodes} episodes: "
                            f"{eval_score:.3f} "
                        )
                        print(
                            f"Success percentage over {self.config.n_episodes} episodes: "
                            f"{success:.3f} "
                        )
                        if "drawer" in self.config.env:
                            print(
                                f"Mean object to target distance over {self.config.n_episodes} episodes: "
                                f"{mean_obj_to_target:.3f} "
                            )
                        
                        if "drawer" in self.config.env:
                            wandb.log(
                                {"eval_score": eval_score, "success_percent":success, "object_to_target_distance":mean_obj_to_target}
                            )
                        else:
                            wandb.log(
                                {"eval_score": eval_score, "success_percent":success}
                            )
                        # whether to calculate fid

                        # data = 
                        # if exists(self.inception_v3):
                        #     fid_score = self.fid_score(real_samples = data, fake_samples = all_images)
                        #     accelerator.print(f'fid_score: {fid_score}')

                pbar.update(1)

        accelerator.print('training complete')
    
    
    @torch.no_grad()   
    def eval_actor(self, n_episodes: int, device: str, env_name: str, step: int):
        self.env.seed(self.seed)
        episode_rewards = []
        obj_to_target = 0.0
        success = 0
        save_gif_dir = os.path.join(self.results_folder, 'eval_gifs')
        if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)

        for i in range(n_episodes):
            images = []
            states = []
            state, done = self.env.reset(), False
            episode_reward = 0.0

            images.append(render(self.env, env_name))
            states.append(state)
            while not done:
                
                if len(images) > 1:
                    action = self.act(images[-2 : ], states[-2: ])  # Generate action using both rendered image and low-level obs
                else:
                    action = self.act([images[-1], images[-1]], [states[-1], states[-1]])  # First time step condition on the initial state twice
                action = action[0]
                try:
                    state, reward, done, extra = self.env.step(action)
                except:
                    state, reward, terminated, truncated, extra = self.env.step(action)
                    done = terminated or truncated

                episode_reward += reward
                images.append(render(self.env, env_name))
                states.append(state)

                if "drawer" in env_name and int(extra["success"]) == 1:
                    success += 1
                    obj_to_target += extra["obj_to_target"]
                    break
                elif done:
                    success += 1
                    break

            if int(extra["success"]) != 1 and "drawer" in env_name:
                obj_to_target += extra["obj_to_target"]
            
            episode_rewards.append(episode_reward)
            save_gif_path = os.path.join(save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(step, i, round(episode_reward, 2)))
            utils_c.save_numpy_as_gif(np.array(images), save_gif_path)

        success_rate = float(success) / float(n_episodes)
        obj_to_target_avg = obj_to_target / float(n_episodes)

        if "drawer" in env_name:
            return np.asarray(episode_rewards), success_rate, obj_to_target_avg
        else:
            return np.asarray(episode_rewards), success_rate

    def act(self, images, obs):
        device = self.device
        bs = 1
        # print(images[0].shape)
        x_conds = torch.Tensor(np.array([images[0], images[1]])).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        for o in obs:
            o = (o - self.obs_mean) / self.obs_std
        obs = torch.Tensor(np.array([obs[0], obs[1]])).unsqueeze(0).to(device)
        # tasks = self.encode_batch_text(tasks).to(device)

        with self.accelerator.autocast():
            output = self.ema.ema_model.sample(batch_size=bs, x_cond=x_conds, obs=obs)
        output = output.cpu().numpy().squeeze(0)
        output = output * self.actions_std + self.actions_mean
        # print("output : ", output.shape)
        return output
