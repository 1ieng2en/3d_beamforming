from cgi import print_arguments
from SoundFieldAnalysis import SoundFieldAnalysis
import open3d as o3d
import os
import glob
from data_importer import PointCloudManager, DataExtractor


# this is the main part of calculation, pcd input should be aligned first in pcd_align.py

base_folder = "postPCD"  # dir to PCD folders
manager = PointCloudManager(base_folder)
pcd_mic, pcd, filtered_pcd = manager.run()

extractor = DataExtractor('vvs-20kHz-12.8k-1mTOback-rightear.mat')
recording = extractor.load_data()

while True:
    freq_input = input("Enter frequency (or 'n' to exit): ")
    if freq_input.lower() == 'n':
        break  

    dynamic_range_input = input("Enter dynamic range (or 'n' to exit): ")
    print("Input done, now calculating")
    print("---------------------------")
    if dynamic_range_input.lower() == 'n':
        break

    try:
        freq = int(freq_input)
        dynamic_range = int(dynamic_range_input)
    except ValueError:
        print("Please enter valid integers for frequency and dynamic range.")
        continue

    BF_analysis = SoundFieldAnalysis(pcd_mic, recording, filtered_pcd, freq=freq)
    BF_analysis.calculate_3D_DAS()
    BF_analysis.plot(dynamic_range=dynamic_range)
    print("finish calculation, close result window to quit or assign new parameters")