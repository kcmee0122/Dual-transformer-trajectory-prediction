# Dual-transformer-trajectory-prediction


---

This codebase implements the system described in the paper:

**Integrated Prediction of Maneuver and Trajectory Based on Multi-task Transformer for Guidance of Multiple Vehicles at Unsignalized Intersections**


---

## Requirements

Building and using requires the following libraries and programs:

- cuda >= 11.7
- python >= 3.11.4
- torch >= 2.0.0+cu117
- pandas >= 2.2.3
- numpy >= 1.26.4
- tqdm >= 4.66.5
- natsort >= 7.1.1
- opencv-python >= 4.10.0
- pillow >= 11.0.0
- imageio >= 2.33.1

```plaintext
pip install -r requirements.txt
```

## Usage

---

### 1. Get Data
You can download the inD data set from the following path:
**Z:\OpenDataset\inD-dataset-v1.0**

#### 1.1 Delete vulnerable road users (VRUs) and parked vehicles

1.1.1 Setting

- Create a folder named **raw**, and place the inD-dataset inside it.

1.1.2 Run Get Data Code

- Running the following code will remove the VRU (Vulnerable Road Users) and parked vehicles from the inD dataset.

```plaintext
python Del_vru_static.py
```
---

#### 1.2 Annotation maneuver

1.2.1 Run Get Data Code

- Running the following code will annotate the maneuvers in the inD dataset.

```plaintext
python Annotation_inD_siteB.py
python Annotation_inD_siteD.py
```


### 2.Train

#### 2.1 Split training and validation set

2.1.1 Setting

- split_ratio : train/validation split ratio based on the number of vehicles in a CSV file from the inD dataset

```plaintext
split_ratio = (7, 3) 
```

2.1.2 Run Train Code

- Running the following code will load a CSV file from the inD dataset and split it into training and validation sets based on the number of vehicles.

```plaintext
python Split_train_vali.py
```

#### 2.2 run train code
2.2.1 Setting

- site : one of the inD dataset locations

```plaintext
site = "site_B"
```

2.2.2 Run Train Code

```plaintext
python Train.py
```

---

### 3.Test

#### 3.1 Visualization

3.1.1 Setting

- Data_num :  Recording number of the inD data to be visualized
- Data_name : CSV file of the inD dataset for testing  
- log_dir : result of **2.2 run train code**       
- img_path_name : Folder name to save the image

```plaintext
Data_num      = 18
Data_name     = '../Get_data/preprocessing_maneuver/18_tracks_del_vru_static_maneuver.csv'
log_dir       = r'Z:\kcm\졸업\Code\inD_vehicle_B\log\Performance\Dual_transformer_Config_TEST_25_75_2024-10-16-21-25'
img_path_name = "inD_site_B_visualization" 
```

3.1.2 Run Test Code

```plaintext
python visualization.py
```

#### 3.2 Inference
3.2.1 Setting 

- Data_dir : CSV file(s) of the inD dataset directory for testing  
- log_dir : result of **2.2 run train code**       

```plaintext
Data_dir : r'Z:\kcm\졸업\Code\inD_vehicle_B\Data\VALI'  
log_dir : r'Z:\kcm\졸업\Code\inD_vehicle_B\log\Performance\Dual_transformer_Config_TEST_25_75_2024-10-16-21-25'       
```

3.2.2 Run Test Code

```plaintext
python inference.py
```
---
