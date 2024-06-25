import d3rlpy
import pickle as pkl
import numpy as np
import uuid

# with open("/home/venky/Desktop/RL-VLM-F/test_dummy/cartpole/data_replay_pred.pkl", 'rb') as f:
#    data= pkl.load(f)
# from envs.cartpole import CartPoleEnv
# def make_numpy(data:dict):

#     data["observations"] = np.array(data["observations"])
#     # data["images"] = np.array(data["images"])
#     data["actions"] = np.array(data["actions"])
#     data["next_observations"] = np.array(data["next_observations"])
#     data["next images"] = np.array(data["next images"])
#     data["rewards"] = np.array(data["rewards"])
#     data["terminals"] = np.array(data["terminals"])
#     return data
 
# env = CartPoleEnv()
# data = make_numpy(data) 
data, env = d3rlpy.datasets.get_cartpole(dataset_type='replay')
 
exp_name = "DQN-Cartpole-d3rlpy"    +  str(uuid.uuid4())
dqn = d3rlpy.algos.DQNConfig().create()
file_logger = d3rlpy.logging.FileAdapterFactory(root_dir="/home/venky/Desktop/RL-VLM-F/offline_rl_exp/cartpole/d3rlpy_logs")
wandb_logger = d3rlpy.logging.WanDBAdapterFactory(project="RISS").create(exp_name)
# logger = d3rlpy.logging.utils.CombineAdapterFactory([file_logger,wandb_logger])
print("starting Training")
dqn.fit(
   dataset=data,
   n_steps=100000,
   # set FileAdapterFactory to save metrics as CSV files
   logger_adapter=wandb_logger,
   evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
)