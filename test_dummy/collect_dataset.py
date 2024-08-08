from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

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
def save_frame(path, frame):
    imageio.imwrite(path, frame)


collection_config = {
    "demos": 25,
    "output_path": "/home/venky/Desktop/RL-VLM-F/test_dummy",
    "resolution": (300, 300),
    "safety": 0, ### discard the last {ratio} of the collected videos (preventing failed episodes)
}


included_tasks = ["drawer-open"]
included_tasks = [t + "-v2-goal-observable" for t in included_tasks]

def collect_trajectory(init_obs, env, env_name, policy, seed, image_height=300, image_width=300, epsilon = 0.1, image_reward=True):
    images = []
    next_images = []
    actions = []
    state = []
    next_state = []
    rewards = []
    terminals = []
    info = []
    
    episode_return = 0
    done = False
    obs = init_obs
    
    if "metaworld" in env_name:
        # print(inspect.getargspec(env.render))
        # env.render_mode = "rgb_array"
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

    if image_reward and \
        'Water' not in env_name and \
            'Rope' not in env_name:
        image = cv2.resize(rgb_image, (image_height, image_width)) # NOTE: resize image here
    
    
    while not done:
        images += [image]
        state += [obs]
        
        rand =  np.random.uniform(low=0.0, high=1.0)
        if rand<epsilon:
            action = env.action_space.sample()
            # print(action)
        else:
            action = policy.get_action(obs)
            #action_1 = env.action_space.sample()
            #print(action.shape, action_1.shape)
        try:
            try: # for handle stupid gym wrapper change 
                next_obs, reward, done, extra = env.step(action)
                # print("Here")
            except:
                next_obs, reward, terminated, truncated, extra = env.step(action)
                done = terminated or truncated
                
                # print(terminated)
                # print("HERE\n")
        except Exception as e:
            print(e)
            break
        
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

        if image_reward and \
            'Water' not in env_name and \
                'Rope' not in env_name:
            image = cv2.resize(rgb_image, (image_height, image_width)) # NOTE: resize image here
    
        
        next_images+=[image]
        actions += [action]
        rewards +=[reward]
        next_state +=[next_obs]
        terminals +=[done]
        info += [extra]
        episode_return += reward
        if int(extra["success"]) == 1:
            break
        obs = next_obs
       
        
    # print("\n")   
    # print(len(state), len(actions), len(rewards), len(next_state),len(next_images))   
    # print("\n")
    demo_dir = "/home/venky/Desktop/RL-VLM-F/test_dummy/drawer-open/42/" + str(seed) + "/"
    os.makedirs(demo_dir, exist_ok=True)
    for i, frame in enumerate(images):
        with mp.Pool(10) as p:
            p.starmap(save_frame, [(os.path.join(demo_dir, f"{i:03d}.png"), frame)])
    # cv2.imshow(' image',images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("saved sequence")
    return state, images, actions, next_state, next_images, rewards, episode_return, terminals, info  


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
    
ps = {}
for env_name in env_dict.keys():
    policy = get_policy(env_name)
    if policy is None:
        print("Policy not found:", env_name)
    else:
        ps[env_name] = policy

out_path = collection_config["output_path"]

data = {}
data["observations"] = []
data["images"] = []
data["actions"] = []
data["next_observations"] = []
data["next images"] = []
data["rewards"] = []
data["terminals"] = []
data["info"] = []
print("inits done")

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
        state, images, actions, next_state, next_images, rewards, episode_return, terminals, info = collect_trajectory(obs, env, env_name, ps[task],seed, epsilon=1.00)
        print("data collected for seed:", seed)
        # assert len(images) == len(action_seq) + 1 or len(images) == 502
        data["observations"] += state
        data["images"] += images
        data["actions"] += actions
        data["next_observations"] += next_state
        data["next images"] += next_images
        data["rewards"] += rewards
        data["terminals"] += terminals
        data["info"] += info
    data["observations"] = np.array(data["observations"])
    data["images"] = np.array(data["images"])
    data["actions"] = np.array(data["actions"])
    data["next_observations"] = np.array(data["next_observations"])
    data["next images"] = np.array(data["next images"])
    data["rewards"] = np.array(data["rewards"])
    data["terminals"] = np.array(data["terminals"])
    data["info"] = np.array(data["info"])
    ### save the collected demos

    with open(f"{demo_dir}/data.pkl", "wb") as f:
        pickle.dump(data, f)

print("Completed data collection for all tasks")
        

        
    
    
        
        