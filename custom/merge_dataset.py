import numpy as np
import pickle
import os

def combine_data_from_directory(dataset_dir):
    # Initialize a dictionary to store combined data
    data_dict = {"observations": [], "actions": [], "next_observations": [], "rewards": [], "terminals": []}

    # Iterate through each file in the dataset directory
    print("List of files in the directory", os.listdir(dataset_dir))
    for file_name in os.listdir(dataset_dir):
        if file_name.startswith('data_') and file_name.endswith('.0.pkl'):
            file_path = os.path.join(dataset_dir, file_name)
            print(f'Loading data from {file_path}')
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                # Append the data from each file into the corresponding lists in data_dict
                data_dict["observations"].append(data["observations"])
                data_dict["actions"].append(data["actions"])
                data_dict["next_observations"].append(data["next_observations"])
                data_dict["rewards"].append(data["rewards"])
                data_dict["terminals"].append(data["terminals"])
                data_dict["rewards_pred"].append(data["rewards_pred"])  
                del data

    # Concatenate the lists into numpy arrays
    data_dict["observations"] = np.concatenate(data_dict["observations"])
    data_dict["actions"] = np.concatenate(data_dict["actions"])
    data_dict["next_observations"] = np.concatenate(data_dict["next_observations"])
    data_dict["rewards"] = np.concatenate(data_dict["rewards"])
    data_dict["terminals"] = np.concatenate(data_dict["terminals"])
    data_dict["rewards_pred"] = np.concatenate(data_dict["rewards_pred"])

    # Save the combined data to a new pickle file
    combined_file_path = os.path.join(os.path.join(os.getcwd(), "exp"), 'ropeflatteneasy-random.pkl')
    with open(combined_file_path, 'wb') as file:
        pickle.dump(data_dict, file)
    print(f'Combined data saved to {combined_file_path}')
    return combined_file_path

dataset_directory = '/mnt/sda1/sreyas/RL_VLM_F-exp/datagen/Rope_flatten_easy/datagen_RopeFlattenEasy-Random/softgym_RopeFlattenEasy/2024-08-31-13-00-31/vlm_1gemini_free_form_rewardlearn_from_preference_H256_L3_lr0.0003/teacher_b-1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/datagen_PassWater_init1000_unsup9000_inter5000_maxfeed20000_seg1_acttanh_Rlr0.0001_Rbatch100_Rupdate30_en3_sample0_large_batch10_seed0'
combined_data_file = combine_data_from_directory(dataset_directory)
print(f'Combined data saved to {combined_data_file}')