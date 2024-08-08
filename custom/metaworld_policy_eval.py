from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import pyrallis
import gym
import metaworld.policies as policies
from math import ceil
from tqdm import tqdm
import numpy as np
import imageio
import os
import pickle
import multiprocessing as mp
import cv2
import imageio
import inspect
from dataclasses import asdict, dataclass
import utils
import wandb
import uuid



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

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "metaworld_drawer-open-v2"  # OpenAI gym environment name
    n_episodes: int = 10
    # Wandb logging
    project: str = "RISS"
    group: str = "IQL-debug"
    name: str = "IQL-debug-eval-50-expert_policy-c"

    
def wandb_init(config: dict) -> None:
    wandb.init(
        #mode="disabled",
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
    
def save_frame(path, frame):
    imageio.imwrite(path, frame)


def collect_trajectory( env, actor, device: str, n_episodes: int, seed: int, env_name :str, save_gif_dir:str, step:int
) :
    env.seed(seed)
    episode_rewards = []
    obj_to_target = 0.0
    success = 0

    if not os.path.exists(save_gif_dir):
            os.makedirs(save_gif_dir)
    for i in range(n_episodes):
        images = []
        state, done = env.reset(), False
        episode_reward = 0.0
        if "metaworld" in env_name:
            state = state[0]
        images.append(render(env,env_name))
        # if True:
        #     image = render(env, env_name)
        #     cv2.imshow("image",image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        while not done:
            action = actor.get_action(state)
            # state, reward, done, _ = env.step(action)
            try: # for handle stupid gym wrapper change 
                state, reward, done, extra = env.step(action)
                # print("Here")
            except:
                state, reward, terminated, truncated, extra = env.step(action)
                done = terminated or truncated
            # if True:
            #     image = render(env, env_name)
            #     cv2.imshow("image",image)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            episode_reward += reward
            images.append(render(env,env_name))
            if "drawer" in env_name:
                 if int(extra["success"]) == 1:
                    success = success + 1
                    obj_to_target = obj_to_target + extra["obj_to_target"]
                    break
            else:
                if done: 
                    success = success + 1
                    break
        if int(extra["success"]) != 1:
            obj_to_target = obj_to_target + extra["obj_to_target"]
        episode_rewards.append(episode_reward)
        save_gif_path = os.path.join(save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(step, i, round(episode_reward, 2)))
        utils.save_numpy_as_gif(np.array(images), save_gif_path)
        
    success = float(success)
    success = success/float(n_episodes)
    obj_to_target = obj_to_target/float(n_episodes)
    return np.asarray(episode_rewards), success, obj_to_target


def get_policy(env_name):
    name = "".join(" ".join(env_name.split('-')[:-3]).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

def save_frame(path, frame):
    imageio.imwrite(path, frame)

@pyrallis.wrap()
def eval_metaworld(config:TrainConfig):
    collection_config = {
        "demos": 25,
        "output_path": "/home/venky/Desktop/RL-VLM-F/test_dummy",
        "resolution": (300, 300),
        "safety": 0, ### discard the last {ratio} of the collected videos (preventing failed episodes)
    }

    wandb_init(asdict(config))
    included_tasks = ["drawer-open"]
    included_tasks = [t + "-v2-goal-observable" for t in included_tasks]
    #included_tasks = ['assembly-v2-goal-observable', 'basketball-v2-goal-observable', 'bin-picking-v2-goal-observable', 'box-close-v2-goal-observable', 'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable', 'button-press-v2-goal-observable', 'button-press-wall-v2-goal-observable', 'coffee-button-v2-goal-observable', 'coffee-pull-v2-goal-observable', 'coffee-push-v2-goal-observable', 'dial-turn-v2-goal-observable', 'disassemble-v2-goal-observable', 'door-close-v2-goal-observable', 'door-lock-v2-goal-observable', 'door-open-v2-goal-observable', 'door-unlock-v2-goal-observable', 'hand-insert-v2-goal-observable', 'drawer-close-v2-goal-observable', 'drawer-open-v2-goal-observable', 'faucet-open-v2-goal-observable', 'faucet-close-v2-goal-observable', 'hammer-v2-goal-observable', 'handle-press-side-v2-goal-observable', 'handle-press-v2-goal-observable', 'handle-pull-side-v2-goal-observable', 'handle-pull-v2-goal-observable', 'lever-pull-v2-goal-observable', 'pick-place-wall-v2-goal-observable', 'pick-out-of-hole-v2-goal-observable', 'reach-v2-goal-observable', 'push-back-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable', 'plate-slide-v2-goal-observable', 'plate-slide-side-v2-goal-observable', 'plate-slide-back-v2-goal-observable', 'plate-slide-back-side-v2-goal-observable', 'peg-unplug-side-v2-goal-observable', 'soccer-v2-goal-observable', 'stick-push-v2-goal-observable', 'stick-pull-v2-goal-observable', 'push-wall-v2-goal-observable', 'reach-wall-v2-goal-observable', 'shelf-place-v2-goal-observable', 'sweep-into-v2-goal-observable', 'sweep-v2-goal-observable', 'window-open-v2-goal-observable', 'window-close-v2-goal-observable']

    ps = {}
    for env_name in env_dict.keys():
        policy = get_policy(env_name)
        if policy is None:
            print("Policy not found:", env_name)
        else:
            ps[env_name] = policy

    out_path = collection_config["output_path"]


    os.makedirs(out_path, exist_ok=True)
    for task in tqdm(included_tasks):
        print(task)
        out_dir = os.path.join(out_path, "-".join(task.split('-')[:-3]))
        os.makedirs(out_dir, exist_ok=True)
        demo_dir = os.path.join(out_dir, "42")
        os.makedirs(demo_dir, exist_ok=True)
        for seed in tqdm(range(42, 42+ceil(collection_config["demos"] * (1+collection_config["safety"])))):
            env_name = "drawer-open-v2"
            if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
                env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
            else:
                env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
            
            env = env_cls(render_mode='rgb_array')
            env.camera_name = env_name
            
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.seed(seed=seed)
            #env = env_dict[task](seed=seed)
            
            # print(env.observation_space.shape, env1.observation_space.shape)
            
            env_name = "metaworld_" + env_name
            obs = env.reset()
            # print(len(obs))
            if "metaworld" in env_name:
                obs = obs[0]
            eval_scores, success, mean_obj_to_target = collect_trajectory(env, ps[task],"cuda", config.n_episodes, seed, env_name, save_gif_dir="/home/venky/Desktop/dummy/eval", step=500*(seed-42))
            eval_score = eval_scores.mean()
            #normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            #evaluations.append(eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} "
            )
            print(
                f"Success percentage over {config.n_episodes} episodes: "
                f"{success:.3f} "
            )
            print(
                f"Mean object to target distance over {config.n_episodes} episodes: "
                f"{mean_obj_to_target:.3f} "
            )
            print("---------------------------------------")
            wandb.log(
                {"eval_score": eval_score, "success_percent":success, "object_to_target_distance":mean_obj_to_target}, step=seed)
        



    print("Completed eval  for all tasks")
            
if __name__ == "__main__":
    eval_metaworld()     
    
    
        
        