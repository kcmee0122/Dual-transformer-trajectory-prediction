# Dual-transformer-trajectory-prediction

# Requirements
Building and using requires the following libraries and programs:

* cuda >= 11.7
* python >= 3.11.4
* torch >= 2.0.0+cu117
* pandas >= 2.2.3
* numpy >= 1.26.4
* tqdm >= 4.66.5
* natsort >= 7.1.1
* opencv-python >= 4.10.0
* pillow >= 11.0.0
* imageio >= 2.33.1

# Usage
1. Get Data
You can download the inD data set from the following path: Z:\OpenDataset\inD-dataset-v1.0

1.1 Delete vulnerable road users (VRUs) and parked vehicles

1.1.1 Setting
* Create a folder named raw, and place the inD-dataset inside it.

1.1.2 Run Get Data Code
* Running the following code will remove the VRU (Vulnerable Road Users) and parked vehicles from the inD dataset.

1.2 Annotation maneuver

1.2.1 Run Get Data Code
* Running the following code will annotate the maneuvers in the inD dataset.

> python Annotation_inD_siteB.py

2.Train
2.1 Split training and validation set

2.1.1 Setting
* split_ratio : train/validation split ratio based on the number of vehicles in a CSV file from the inD dataset

2.1.2 Run Train Code
* Running the following code will load a CSV file from the inD dataset and split it into training and validation sets based on the number of vehicles.

2.2 run train code

2.2.1 Setting
* site : one of the inD dataset locations

2.2.2 Run Train Code

3.Test
3.1 Visualization

3.1.1 Setting

* Data_num : Recording number of the inD data to be visualized
* Data_name : CSV file of the inD dataset for testing
* log_dir : result of 2.2 run train code
* img_path_name : Folder name to save the image

3.1.2 Run Test Code

3.2 Inference

3.2.1 Setting

* Data_dir : CSV file(s) of the inD dataset directory for testing
* log_dir : result of 2.2 run train code

3.2.2 Run Test Code
