import d3rlpy
from envs import cartpole
import cv2
import numpy as np
from tqdm import tqdm
import os
import pickle as pkl

dataset, env = d3rlpy.datasets.get_cartpole(dataset_type='replay')
env = cartpole.CartPoleEnv()

# Print the action space details
print(f"Action space: {env.action_space}")
print(dataset)
data = {}
data["observations"] = []
# data["images"] = []
data["actions"] = []
data["next_observations"] = []
data["next images"] = []
data["rewards"] = []
data["terminals"] = []

episodes = dataset.episodes
print(episodes[0])
for episode in tqdm(episodes):
    for transition in episode.transitions:
        action = transition.action.astype(np.float32)
        if action == 0:
            action = -1
        # Ensure the action is within bounds
        # print(f"Original Action: {action}")
        if not env.action_space.contains(action):
            action = np.clip(action, env.action_space.low, env.action_space.high)
            #print(f"Clipped Action: {action}")

        observation = transition.observation
        terminal = bool(transition.terminal)

        env.reset_at_state(observation)
        # image = env.render(mode='rgb_array')
        next_observation, reward, terminated, truncated, extra = env.step(action)
        next_image = env.render(mode='rgb_array')

        data["observations"].append(observation)
        data["actions"].append(action)
        data["next_observations"].append(next_observation)
        data["next images"].append(next_image)
        data["rewards"].append(reward)
        data["terminals"].append(terminal)
        # data["images"].append(image)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
# print(len(data["actions"]))
demo_dir = "/home/venky/Desktop/RL-VLM-F/test_dummy/cartpole/"
os.makedirs(demo_dir, exist_ok=True)
print("Saving data")
with open(f"{demo_dir}/data_replay_corrected.pkl", "wb") as f:
    pkl.dump(data, f)
print("saved data at ", demo_dir)

exit()