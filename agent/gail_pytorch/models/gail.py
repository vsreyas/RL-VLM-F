import numpy as np
import torch
import os
from torch.nn import Module

from agent.gail_pytorch.models.nets import PolicyNetwork, ValueNetwork, Discriminator
from agent.gail_pytorch.utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor
import wandb
import cv2
import utils
from tqdm import tqdm

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

class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)

        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.mean.detach().cpu().numpy()

        return action
    
    def save(self, step):
        ckpt_path = self.train_config.checkpoints_path
        torch.save(
            self.pi.state_dict(), os.path.join(ckpt_path, f"policy_{step}.ckpt")
        )
        torch.save(
            self.v.state_dict(), os.path.join(ckpt_path, f"value_{step}.ckpt")
        )
        torch.save(
            self.d.state_dict(), os.path.join(ckpt_path, f"discriminator_{step}.ckpt")
        )
        print("Model saved at step: ", step)
        
    def load(self, step):
        ckpt_path = self.train_config.checkpoints_path
        self.pi.load_state_dict(
            torch.load(os.path.join(ckpt_path, f"policy_{step}.ckpt"))
        )
        self.v.load_state_dict(
            torch.load(os.path.join(ckpt_path, f"value_{step}.ckpt"))
        )
        self.d.load_state_dict(
            torch.load(os.path.join(ckpt_path, f"discriminator_{step}.ckpt"))
        )
        print("Model loaded from step: ", step)

    def train(self, env, env_name, seed, n_episodes, obs_, acts_, render_=False):
        num_iters = self.train_config.num_iters
        # num_steps_per_iter = self.train_config.num_steps_per_iter
        horizon = self.train_config.horizon
        lambda_ = self.train_config.lambda_
        gae_gamma = self.train_config.gae_gamma
        gae_lambda = self.train_config.gae_lambda
        eps = self.train_config.epsilon
        max_kl = self.train_config.max_kl
        cg_damping = self.train_config.cg_damping
        normalize_advantage = self.train_config.normalize_advantage

        opt_d = torch.optim.Adam(self.d.parameters(), self.train_config.lr)
        save_gif_dir = self.train_config.gif_path
        
        rwd_iter_means = []
        num_samples = self.train_config.num_samples
        
        obs_ = FloatTensor(obs_)
        acts_ = FloatTensor(acts_)
        first_index = 0
        last_index =  num_samples
            
        if first_index >= len(obs_):
            first_index = 0
            last_index = num_samples
        if last_index + num_samples > len(obs_):
            last_index = len(obs_)
        exp_acts = acts_[first_index:last_index]
        exp_obs = obs_[first_index:last_index]
        first_index = last_index
        last_index = last_index + num_samples
        for i in tqdm(range(num_iters)):
            
            
            for k in range(self.train_config.policy_update_freq):

                rwd_iter = []

                obs = []
                acts = []
                rets = []
                advs = []
                gms = []

                steps = 0
                env.seed(seed)
                obj_to_target = 0.0
                success = 0
                for j in range(n_episodes):
                    ep_obs = []
                    ep_acts = []
                    ep_rwds = []
                    ep_costs = []
                    ep_disc_costs = []
                    ep_gms = []
                    ep_lmbs = []
                    ep_img = []

                    t = 0
                    

                    ob, done = env.reset(), False

                    while not done :
                        act = self.act(ob)

                        ep_obs.append(ob)
                        obs.append(ob)

                        ep_acts.append(act)
                        acts.append(act)
                        if render_ and j == n_episodes -1:
                            ep_img.append(render(env,env_name))

                        try: # for handle stupid gym wrapper change 
                            ob, rwd, done, info = env.step(act)
                            # print("Here")
                        except:
                            ob, rwd, terminated, truncated, info = env.step(act)
                            done = terminated or truncated
                    

                        ep_rwds.append(rwd)
                        ep_gms.append(gae_gamma ** t)
                        ep_lmbs.append(gae_lambda ** t)

                        t += 1
                        steps += 1

                        if horizon is not None:
                            if t >= horizon:
                                done = True
                                break
                            
                        if "drawer" in env_name:
                            if int(info["success"]) == 1:
                                obj_to_target = obj_to_target + info["obj_to_target"]
                                success = success + 1
                                rwd_iter.append(np.sum(ep_rwds))
                                break

                    if int(info["success"]) != 1 and "drawer" in env_name:
                        obj_to_target = obj_to_target + info["obj_to_target"]
                        rwd_iter.append(np.sum(ep_rwds))
                    elif done:
                        print("Here")
                        success = success + 1
                        rwd_iter.append(np.sum(ep_rwds))

                    if render_ and j == n_episodes -1:
                        save_gif_path = os.path.join(save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(i, j, round(np.sum(ep_rwds), 2)))
                        utils.save_numpy_as_gif(np.array(ep_img), save_gif_path)
                        
                    ep_obs = FloatTensor(np.array(ep_obs))
                    ep_acts = FloatTensor(np.array(ep_acts))
                    ep_rwds = FloatTensor(ep_rwds)
                    # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                    ep_gms = FloatTensor(ep_gms)
                    ep_lmbs = FloatTensor(ep_lmbs)

                    ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                        .squeeze().detach()
                    ep_disc_costs = ep_gms * ep_costs

                    ep_disc_rets = FloatTensor(
                        [sum(ep_disc_costs[i:]) for i in range(t)]
                    )
                    ep_rets = ep_disc_rets / ep_gms

                    rets.append(ep_rets)

                    self.v.eval()
                    curr_vals = self.v(ep_obs).detach()
                    next_vals = torch.cat(
                        (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                    ).detach()
                    ep_deltas = ep_costs.unsqueeze(-1)\
                        + gae_gamma * next_vals\
                        - curr_vals

                    ep_advs = FloatTensor([
                        ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                        .sum()
                        for j in range(t)
                    ])
                    advs.append(ep_advs)

                    gms.append(ep_gms)
                success = float(success)
                success = success/float(n_episodes)
                obj_to_target = obj_to_target/float(n_episodes)
                
                rwd_iter_means.append(np.mean(rwd_iter))
                # print(f"Iterations:{i+1}, Rewards:{rwd_iter}")
                print(
                    "Iterations: {},   Reward Mean: {}"
                    .format(i + 1, np.mean(rwd_iter))
                )
                if "drawer" in env_name:
                    wandb.log(
                        {"eval_score": np.mean(rwd_iter), "success_percent":success, "object_to_target_distance":obj_to_target}
                    )
                else:
                    wandb.log(
                        {"eval_score": np.mean(rwd_iter), "success_percent":success}
                    )

                obs = FloatTensor(np.array(obs))
                acts = FloatTensor(np.array(acts))
                rets = torch.cat(rets)
                advs = torch.cat(advs)
                gms = torch.cat(gms)

                if normalize_advantage:
                    advs = (advs - advs.mean()) / advs.std()

                self.d.train()
                exp_scores = self.d.get_logits(exp_obs, exp_acts)
                nov_scores = self.d.get_logits(obs, acts)

                opt_d.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    exp_scores, torch.zeros_like(exp_scores)
                ) \
                    + torch.nn.functional.binary_cross_entropy_with_logits(
                        nov_scores, torch.ones_like(nov_scores)
                    )
                wandb.log({"loss/discriminator": loss})
                loss.backward()
                opt_d.step()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(
                disc_causal_entropy, self.pi
            )
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)
            if i % self.train_config.eval_freq == 0 and i != 0:
                self.save(i)

        return rwd_iter_means
