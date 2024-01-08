
import os
import glob
import open3d as o3d
import scipy.io
import numpy as np

class PointCloudManager:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.directory = None

    def list_subfolders(self):
        return [f.name for f in os.scandir(self.base_folder) if f.is_dir()]

    def load_models(self, subfolder):
        self.directory = f"{self.base_folder}/{subfolder}"
        model_files = sorted(glob.glob(f"{self.directory}/model_*.ply"))

        if len(model_files) < 3:
            print("Not enough model files found in the folder.")
            return None, None, None

        pcd_mic = o3d.io.read_point_cloud(model_files[0])
        pcd = o3d.io.read_point_cloud(model_files[1])
        filtered_pcd = o3d.io.read_point_cloud(model_files[2])

        return pcd_mic, pcd, filtered_pcd

    def run(self, index = None):
        subfolders = self.list_subfolders()

        if not subfolders:
            print("No subfolders found in 'postPCD'.")
            return

        print("Available subfolders:")
        for i, folder in enumerate(subfolders):
            print(f"{i}: {folder}")

        if index == None:    
            choice = int(input("Enter the number of the measurement you want to load: "))
        else:
            choice = index
        selected_subfolder = subfolders[choice]

        return self.load_models(selected_subfolder)

class DataExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        # Load the measured field pressure data
        mat_data = scipy.io.loadmat(self.file_path)

        # Create an empty list to store the extracted data
        data_list = []

        # To ensure data is retrieved in channel order, construct the key names using numbers in the range
        for i in range(1, 37):  # From 1 to 36
            key = f'Channel_{i}_Data'  # Constructing the key name
            if key in mat_data:  # Check if the key exists in the dictionary
                data_list.append(mat_data[key].T)  # If it exists, add the data to the list

        # Convert the list to a matrix using numpy.vstack()
        fieldPressure_r = np.vstack(data_list)
        return fieldPressure_r
