import cv2
import numpy as np
import torch
import os
import time
import pickle as pkl
import multiprocessing as mp
import imageio
import copy
def save_frame(path, frame):
    imageio.imwrite(path, frame)

def read_pkl():
    with open("/home/venky/Desktop/RL-VLM-F/test_dummy/drawer-open/42/data.pkl", 'rb') as f:
                data = pkl.load(f)
    
    images = data["images"]
    
    data_offl = {}
    
    for key in data.keys():
        no_transitions = 0
        data_offl[key] = []
        for i in range(len(data[key])):
            data_offl[key].extend(data[key][i])
            no_transitions = no_transitions + len(data[key][i])
    
    print(len(data_offl["images"]),"\ntransitions: ", no_transitions )
    
    #images = data_offl["images"]
    # demo_dir = "/home/venky/Desktop/RL-VLM-F/test_dummy/drawer-open/42/"  + "all/"
    # os.makedirs(demo_dir, exist_ok=True)
    # for i, frame in enumerate(images):
    #     with mp.Pool(10) as p:
    #         p.starmap(save_frame, [(os.path.join(demo_dir, f"{i:03d}.png"), frame)])
            
    demo_dir = "/home/venky/Desktop/RL-VLM-F/test_dummy/drawer-open/42"
    with open(f"{demo_dir}/data_offl.pkl", "wb") as f:
        pkl.dump(data_offl, f)

def add_done():
    with open("/home/venky/Desktop/RL-VLM-F/test_dummy/drawer-open/42/data.pkl", 'rb') as f:
        data_b = pkl.load(f)
    with open("/home/venky/Desktop/RL-VLM-F/test_dummy/drawer-open/42/data_offl.pkl", "rb") as f:
        data_offl = pkl.load(f)
    data ={}
    data["observations"] = copy.deepcopy(data_offl["state"])
    data["images"] = copy.deepcopy(data_offl["images"])
    data["actions"] = copy.deepcopy(data_offl["actions"])
    data["next observations"] = copy.deepcopy(data_offl["next state"])
    data["next images"] = copy.deepcopy(data_offl["next images"])
    data["rewards"] = copy.deepcopy(data_offl["rewards"])
    data["terminals"] = []
    
    for j in range(len(data_b["state"])):
        for i in range(len(data_b["state"][j])):
            data["terminals"].append(False)
        data["terminals"][-1] = True
    
    demo_dir = "/home/venky/Desktop/RL-VLM-F/test_dummy/drawer-open/42"
    with open(f"{demo_dir}/data_d4rl.pkl", "wb") as f:
        pkl.dump(data, f)
    print(data.keys())
    for key in data.keys():
        print(key, ": ",len(data[key]),"\n")

def make_numpy():
    with open("/home/venky/Desktop/RL-VLM-F/test_dummy/cartpole/data_replay_pred.pkl", 'rb') as f:
        data= pkl.load(f)
    data["observations"] = np.array(data["observations"])
    # data["images"] = np.array(data["images"])
    data["actions"] = np.array(data["actions"])
    data["next_observations"] = np.array(data["next_observations"])
    data["next images"] = np.array(data["next images"])
    data["rewards"] = np.array(data["rewards"])
    data["terminals"] = np.array(data["terminals"])
    demo_dir = "/home/venky/Desktop/RL-VLM-F/test_dummy/cartpole/"
    print("saving data")
    with open(f"{demo_dir}/data_replay_pred_numpy.pkl", "wb") as f:
        pkl.dump(data, f)
    print("saved")
    
def check():
    with open("/home/venky/Desktop/RL-VLM-F/test_dummy/cartpole/data_replay.pkl", 'rb') as f:
        data= pkl.load(f)
        
    print(len(data["actions"]))
make_numpy()