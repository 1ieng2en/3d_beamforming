# Project Description

## Overview
This project aims to develop a 3D beamforming system for audio applications. Beamforming is a signal processing technique used to enhance the directivity of a microphone array, allowing for improved sound capture and noise reduction. The goal of this project is to implement a real-time beamforming algorithm that can accurately localize and enhance sound sources in three-dimensional space.

## Key Features
- 3D sound source localization: The system will be able to accurately determine the direction and position of sound sources in three-dimensional space.
- Adaptive beamforming: The beamforming algorithm will dynamically adjust the beamforming weights based on the location and characteristics of the sound sources.

## Technologies Used
- Programming language: Python, and Jupyter notebook
- Audio processing libraries: Acoular
- Hardware requirements: Microsoft Azure Kinect and B&K acoustic camera.

## Code Navigation
### 3dbf_app_simulation.ipynb
- with bf3d_prep - data_importer - SoundFIeldAnalysis: is the main part of the 3d beamforming reconstruction
- bf3d_prep for point cloud post processing i.e., gradient calculation and reverse sound field calculation.
- data_importer for point cloud import
- SoundFieldAnalysis for beamforming calculation. Part of the calculation were replaced by Acoular since due to it's performance.
### pcd_aligner_app.ipynb
- with pcd_aligner is for pcd alignment
### simulation.ipynb
- for simulation, in the idealized simulated environment.
### Measurement_normal.ipynb and Measurement.ipynb
- for eq curve, the first one somehow works better than the second one, but the second one has a better plotting. This two are almost identical.
### environment.yml
- for jupyter lab environment
### Kinfuscan.py/Main.py/manual_icp.py/pcd_align.py
- is not activly used now

## Project Goals
- Develop a robust and efficient 3D beamforming algorithm.
- Implement a system capable of processing audio signals in three-dimensional space.
- Evaluate the performance of the system through extensive testing and experimentation.

## Getting Started
To get started with this project, follow the instructions below:

1. Clone the repository: `git clone [repository URL]`
2. Install the required dependencies: [Specify the necessary dependencies and installation steps]
3. Run the project: [Provide instructions on how to run the project]

## Contributing
Contributions to this project are welcome. If you would like to contribute, please follow the guidelines outlined in [CONTRIBUTING.md].

## License
This project is licensed under the [Specify the license you choose for your project].
