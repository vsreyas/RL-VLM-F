#!/bin/bash

# Define the four --fl argument values
fl_values=("0-25" "0-50" "0-75" "prox_flip")
seeds=(42 43 44)
# List your available GPU IDs; adjust this array to match your system
gpus=(1 2 3)
gpu_count=${#gpus[@]}

task_index=0

# Loop over expert dataset runs and assign each run a GPU in a round-robin manner
for fl in "${fl_values[@]}"; do
    for seed in "${seeds[@]}"; do
        gpu_index=$(( task_index % gpu_count ))
        gpu=${gpus[$gpu_index]}
        echo "Launching seed $seed with --fl $fl on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python agent/iql.py \
            --seed "$seed" \
            --data_set_path "/project_data/held/sreyas/RL-VLM-F/data/Drawer_open/Drawer_open-medium-NL.pkl" \
            --project "NL-Drawer_open-medium" \
            --env "metaworld_drawer-open-v2" \
            --vf_lr 3e-5\
            --qf_lr 3e-5\
            --actor_lr 3e-5\
            --nl True \
            --fl "$fl" &
        (( task_index++ ))
    done
done

# Wait for all background processes to finish
wait
