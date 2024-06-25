import cv2
import numpy as np
import torch
import os
import time
import pickle as pkl

# from logger import Logger
# from replay_buffer import ReplayBuffer
# from reward_model import RewardModel
# from reward_model_score import RewardModelScore
# from collections import deque
# from prompt import clip_env_prompts

# import utils
import hydra
# from PIL import Image

# from vlms.blip_infer_2 import blip2_image_text_matching
# from vlms.clip_infer import clip_infer_score as clip_image_text_matching


def make_dataset_from_cached_labels(cfg):
    # if 'metaworld' in cfg.env:
    #     env = utils.make_metaworld_env(cfg)
    # elif cfg.env in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
    #     env = utils.make_classic_control_env(cfg)
    # elif 'softgym' in cfg.env:
    #     env = utils.make_softgym_env(cfg)
    # else:
    #     env = utils.make_env(cfg)
    # ds = env.observation_space.shape[0]
    # da = env.action_space.shape[0]
    
    # cached_label_path = cfg.cached_label_path
    # file_path = os.path.abspath(__file__)
    # dir_path = os.path.dirname(file_path)
    # cached_label_path = "{}/{}".format(dir_path, cached_label_path)
    # read_cache_idx = 0
    # drawer_dataset = {}
    # if cached_label_path is not None:
    #     all_cached_labels = sorted(os.listdir(cached_label_path))
    #     all_cached_labels = [os.path.join(cached_label_path, x) for x in all_cached_labels]

    with open("/home/venky/Desktop/RL-VLM-F/data/cached_labels/Drawer/seed_0/2024-01-22-03-27-22.pkl", 'rb') as f:
                data = pkl.load(f)

    combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
    
    cv2.imshow(' image',combined_images_list[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # image1 = combined_images_list[: , : , :300 , : ]
    # image2 = combined_images_list[: , : , 300: , : ]
    # state_1 = sa_t_1[: , 0 , :ds]
    # action_1 = sa_t_1[: , 0 , ds:]
    # state_2 = sa_t_2[: , 0 , :ds]
    # action_2 = sa_t_2[: , 0 , ds:]
    # r_t_1 = r_t_1[: , 0, : ]
    # r_t_2 = r_t_2[: , 0, : ]
    # print(sa_t_1.shape, r_t_1.shape,state_1.shape, action_1.shape)
    

@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    make_dataset_from_cached_labels(cfg)

if __name__ == '__main__':
    with open("/home/venky/Desktop/RL-VLM-F/data/cached_labels/Drawer/seed_0/2024-01-22-03-27-22.pkl", 'rb') as f:
                data = pkl.load(f)

    combined_images_list, rational_labels, vlm_labels, sa_t_1, sa_t_2, r_t_1, r_t_2 = data
    for img in combined_images_list:
        cv2.imshow(' image',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()