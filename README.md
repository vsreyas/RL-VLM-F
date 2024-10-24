# Offline RL-VLM-F
This is the official codebase for:  
[Real-World Offline Reinforcement Learning from Vision Language Model Feedback](https://vsreyas.github.io/OfflineRL-VLMF-home/),   
Sreyas Venkataraman*, Yufei Wang*, Ziyu Wang, David Held&dagger;, Zackory Erickson&dagger;,   
[Website](https://vsreyas.github.io/OfflineRL-VLMF-home/)

<img width="700px" src="imgs/teaser.gif"/>

## Install
Install the conda env via
```
conda env create -f conda_env.yml
conda activate rlvlmf
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch  
pip install numpy==1.26.0
```

We use customized softgym envs (for cloth fold and straighten rope), as provided in `softgym` folder. Please refer to https://github.com/Xingyu-Lin/softgym?tab=readme-ov-file for compiling softgym. 


## Run experiments
- Get a Gemini api key: follow instructions at https://aistudio.google.com/app/apikey 
- Set the environment variable `export GEMINI_API_KEY=your_obtained_key`.
- Run `source prepare.sh` to prepare some environment variables.    
- Then please see `run.sh` for running experiments with different environments.    
- We use GPT4v for the cloth fold task. Set the environment variable `export OPENAI_API_KEY=your_api_key`, and you should be good to go.   

## Cached VLM preference labels
- Due to that Gemini-pro 1.0 has greatly decreased its free quota to be only 1500 request per day: https://ai.google.dev/pricing, we provide some of the VLM preference labels we cached when running the experiments. We only stored them at an interval during training, e.g., we stored every 25th time when we queried the VLM. Therefore, the total number of cached preferece labels are fewer than the number for the complete run. The labels are also not on-policy, which means they are not generated using the agent's online experience.  
- Still, we find that we are able to get roughly similar performances by using the cached preference labels, for Fold Cloth, Open Drawer, Soccer, CartPole, Straighten Rope, and Pass Water. The performance of Sweep Into with the cached labels is worse compared to the original results in the paper. 
- The cached preference labels can be downloaded through this [google drive link](https://drive.google.com/drive/folders/1dwvu6fhGJOTGRKEfH-pKrtNC6lH6LQHX?usp=sharing).  
- After downloading, put it under `data` so it looks like `data/cached_labels/env_name/different_seed`.  
- The commands in `run.sh` will by default load the cached preference labels; you can use `cached_label_path=None` to not use the cached labels and query the VLM online during training.   
- If you wish to fully reproduce the results in the paper, please train without using the provided cached labels, and generate the VLM preference labels online using the learning agent's online experience. 

## Offline Data Generation
- run_softgym_dataset.sh is a sample script for generating offline data. The data can be generated using custom/collect_dataset_softgym.py.

## Train reward model from cahced labels
- For training model using cached labels run `python train_PEBBLE.py`. run.sh gives a sample script.

## Train offline agents 
- The agents are provided in `agent`. Naive BC, GAIL, Diffusion Policy and IQL are implemented. The hyperparameters can be directly changed in TrainConfig class in the agent file or as command line arguments. Sample scripts are in run_agent_env.sh.
- To run a particular agent - run `python agent/iql.py`

## Acknowledgements
- We thank the authors of PEBBLE and RL-VLM-F for open sourcing their code, which our code is built on: https://github.com/pokaxpoka/B_Pref and https://github.com/yufeiwang63/RL-VLM-F


