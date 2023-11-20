from SoundFieldAnalysis import SoundFieldAnalysis
import open3d as o3d
import os
import glob
from data_importer import PointCloudManager, DataExtractor


base_folder = "postPCD"  # dir to PCD folders
manager = PointCloudManager(base_folder)
pcd_mic, pcd, filtered_pcd = manager.run()

extractor = DataExtractor('vvs-20kHz-12.8k-1mTOback-rightear.mat')
recording = extractor.load_data()

BF_analysis = SoundFieldAnalysis(pcd_mic, recording,filtered_pcd, freq=9500)
BF_analysis.calculate_3D_DAS()
BF_analysis.plot(dynamic_range = 5)