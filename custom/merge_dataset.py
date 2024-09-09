import numpy as np
import pickle
import os
import cv2
import pathlib

def combine_data_from_directory(dataset_dir):
    # Initialize a dictionary to store combined data
    data_dict = {"observations": [], "actions": [], "next_observations": [], "rewards": [], "terminals": [], "rewards_pred": [], "images_path": []}
    dir_ = "/mnt/sda1/sreyas/sim_dataset/"
    os.makedirs(os.path.join(dir_, 'exp'), exist_ok=True)
    # Iterate through each file in the dataset directory
    print("List of files in the directory", os.listdir(dataset_dir))
    
    images_dir = os.path.join(dataset_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    image_counter = 0
    for file_name in os.listdir(dataset_dir):
        if file_name.startswith('data'): #and file_name.endswith('.0.pkl'):
            file_path = os.path.join(dataset_dir, file_name)
            print(f'Loading data from {file_path}')
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                print(data.keys())
                print("Length:",len(data["observations"]))
                # Append the data from each file into the corresponding lists in data_dict
                data_dict["observations"].append(data["observations"])
                data_dict["actions"].append(data["actions"])
                data_dict["next_observations"].append(data["next_observations"])
                data_dict["rewards"].append(data["rewards"])
                data_dict["terminals"].append(data["terminals"])
                data_dict["rewards_pred"].append(data["rewards_pred"])  
                
                image_paths = []
                
                for image in data["images"]:
                    # Create a unique filename for each image using the overall counter
                    image_filename = f'image_{image_counter}.npy'
                    image_path = os.path.join(images_dir, image_filename)
                    
                    # Save the image using Numpy
                    np.save(image_path, image)
                    
                    # Append the relative path to the image_paths list
                    relative_image_path = os.path.relpath(image_path, dataset_dir)
                    image_paths.append(relative_image_path)
                    
                    # Increment the image counter
                    image_counter += 1
                # Append the list of image paths to the data_dict
                data_dict["images_path"].append(image_paths)
                
                
                del data

    # Concatenate the lists into numpy arrays
    data_dict["observations"] = np.concatenate(data_dict["observations"])
    data_dict["actions"] = np.concatenate(data_dict["actions"])
    data_dict["next_observations"] = np.concatenate(data_dict["next_observations"])
    data_dict["rewards"] = np.concatenate(data_dict["rewards"])
    data_dict["terminals"] = np.concatenate(data_dict["terminals"])
    data_dict["rewards_pred"] = np.concatenate(data_dict["rewards_pred"])
    data_dict["images_path"] = sum(data_dict["images_path"], [])
    # Save the combined data to a new pickle file
    
    combined_file_path = os.path.join(dataset_dir, 'PassWater-random.pkl')
    with open(combined_file_path, 'wb') as file:
        pickle.dump(data_dict, file)
    print(f'Combined data saved to {combined_file_path}')
    # print("Data dict images path", data_dict["images_path"])
    return combined_file_path

dataset_directory = '/mnt/sda1/sreyas/RL_VLM_F-exp/datagen/PassWater/datagen_PassWater-Random'
combined_data_file = combine_data_from_directory(dataset_directory)
print(f'Combined data saved to {combined_data_file}')