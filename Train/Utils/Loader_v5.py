import json
import glob
import copy
import torch
import natsort
import numpy as np
import pandas as pd

from Utils.Base import *
from Utils.Count_params import *

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

def Count_trajectory(Data_dir, Config):
    Total_track_count      = 0
    Total_trajectory_count = 0
    
    Data_list = natsort.natsorted(glob.glob(Data_dir))

    for data in tqdm(Data_list):
        Data      = pd.read_csv(data)
        
        if Config['Train_extract'] == False:
            Unique_id = Data['trackId'].unique()
            
        elif Config['Train_extract'] == True:
            Unique_id = np.array([], dtype=np.float32)
            for temp_track_id in Data['trackId'].unique():
                Temp_track_data = Data[Data['trackId']==temp_track_id]
                Temp_track_unique_maneuver = Temp_track_data['Maneuver'].unique()
                if len(Temp_track_unique_maneuver) == 1: 
                    if Temp_track_unique_maneuver == 0:
                        pass
                else:
                    Unique_id = np.concatenate((Unique_id, [temp_track_id]))
                    
        for id in Unique_id:
            Track_data = Data[Data['trackId']==id]
            Track_data_len = len(Track_data) - Config['Input_horizon'] - Config['Prediction_horizon']
            
            if Track_data_len > 0:
                Total_track_count += 1
                for _ in range(0, Track_data_len, Config['Rolling_horizon']):
                    Total_trajectory_count += 1
                    
    print('Track: ', Total_track_count)
    print('Trajectory: ', Total_trajectory_count)
    
    return Total_track_count, Total_trajectory_count

def Count_trajectory_for_vali(Data_dir, Config):
    Total_track_count      = 0
    Total_trajectory_count = 0
    
    Data_list = natsort.natsorted(glob.glob(Data_dir))
    
    for data in tqdm(Data_list):
        Data      = pd.read_csv(data)
        
        if Config['Vali_extract'] == False:
            Unique_id = Data['trackId'].unique()
        elif Config['Vali_extract'] == True:     
            Unique_id = np.array([], dtype=np.float32)
            for temp_track_id in Data['trackId'].unique():
                Temp_track_data = Data[Data['trackId']==temp_track_id]
                Temp_track_unique_maneuver = Temp_track_data['Maneuver'].unique()
                if len(Temp_track_unique_maneuver) == 1: 
                    if Temp_track_unique_maneuver == 0:
                        pass
                else:
                    Unique_id = np.concatenate((Unique_id, [temp_track_id]))
        
        for id in Unique_id:
            Track_data = Data[Data['trackId']==id]
            Track_data_len = len(Track_data) - Config['Input_horizon'] - Config['Prediction_horizon']
            
            if Track_data_len > 0:
                Total_track_count  += 1
                for _ in range(0, Track_data_len, Config['Rolling_horizon']):
                    Total_trajectory_count += 1
                    
    print('Track: ', Total_track_count)
    print('Trajectory: ', Total_trajectory_count)
    
    return Total_track_count, Total_trajectory_count

def Concat_frame_maker(Data_dir, Config):
    Data_list = natsort.natsorted(glob.glob(Data_dir))
    Concat_df = pd.DataFrame()
    
    for index, data in enumerate(Data_list):
        Data           = pd.read_csv(data)
        Data['DataId'] = index
        
        if Config['Train_extract'] == False:
            Concat_df    = pd.concat([Concat_df, Data], axis=0)
        elif Config['Train_extract'] == True:
            Extract_id = np.array([])
            for temp_track_id in Data['trackId'].unique():
                Temp_track_data = Data[Data['trackId']==temp_track_id]
                Temp_track_unique_maneuver = Temp_track_data['Maneuver'].unique()
                if len(Temp_track_unique_maneuver) == 1: 
                    if Temp_track_unique_maneuver == 0:
                        pass
                else:
                    Extract_id = np.concatenate((Extract_id, [temp_track_id]))
                    
            Data = Data[Data['trackId'].isin(Extract_id)].reset_index(drop=True)
            Concat_df    = pd.concat([Concat_df, Data], axis=0)

    Data_columns         = Concat_df.columns        
    Input_feature_index  = np.array([Data_columns.get_loc(feature) for feature in Config['Input_feature']])  # (Config input feature 순서대로 가져오도록 수정)
    Concat_df            = Concat_df.reset_index(drop=True)
    
    Concat_array = Concat_df.iloc[:, Input_feature_index].to_numpy()
    Scaled_concat_array, Concat_mean, Concat_std = standard_scale(Concat_array)

    Scaled_concat_data = Concat_df.loc[:, ['recordingId', 'trackId', 'frame']].copy()
    Scaled_concat_data = pd.concat([Scaled_concat_data, pd.DataFrame(Scaled_concat_array, columns=Config['Input_feature'])], axis=1)
    Scaled_concat_data = pd.concat([Scaled_concat_data, Concat_df.loc[:, ['Maneuver', 'DataId']]], axis=1)
    
    return Scaled_concat_data, Concat_mean, Concat_std

def Concat_frame_maker_for_vali(Data_dir, Config, Train_mean, Train_std):
    Data_list = natsort.natsorted(glob.glob(Data_dir))
    Concat_df = pd.DataFrame()
    
    for index, data in enumerate(Data_list):
        Data           = pd.read_csv(data)
        Data['DataId'] = index

        if Config['Vali_extract'] == False:
            Concat_df    = pd.concat([Concat_df, Data], axis=0)
        elif Config['Vali_extract'] == True:
            Extract_id = np.array([])
            for temp_track_id in Data['trackId'].unique():
                Temp_track_data = Data[Data['trackId']==temp_track_id]
                Temp_track_unique_maneuver = Temp_track_data['Maneuver'].unique()
                if len(Temp_track_unique_maneuver) == 1: 
                    if Temp_track_unique_maneuver == 0:
                        pass
                else:
                    Extract_id = np.concatenate((Extract_id, [temp_track_id]))
                    
            Data = Data[Data['trackId'].isin(Extract_id)].reset_index(drop=True)
            Concat_df    = pd.concat([Concat_df, Data], axis=0)

    Data_columns         = Concat_df.columns        
    Input_feature_index  = np.array([Data_columns.get_loc(feature) for feature in Config['Input_feature']])  # (Config input feature 순서대로 가져오도록 수정)
    Concat_df            = Concat_df.reset_index(drop=True)
    
    Concat_array = Concat_df.iloc[:, Input_feature_index].to_numpy()
    Scaled_concat_array = standard_scale_for_vali(Concat_array, Train_mean, Train_std)

    Scaled_concat_data = Concat_df.loc[:, ['recordingId', 'trackId', 'frame']].copy()
    Scaled_concat_data = pd.concat([Scaled_concat_data, pd.DataFrame(Scaled_concat_array, columns=Config['Input_feature'])], axis=1)
    Scaled_concat_data = pd.concat([Scaled_concat_data, Concat_df.loc[:, ['Maneuver', 'DataId']]], axis=1)
    
    return Scaled_concat_data

def Concat_loader_maker(Concat_data, Trajectory_count, Config, Loader_dir):
    if len(glob.glob(Loader_dir + '/Train_loader_' + str(Config['Input_horizon']) + '_' + str(Config['Prediction_horizon'])+ '.pth')) == 0:
        print('- NO saved loader pth -')
        print('- Generating loader -')
        
        Array_index = 0
        Input_array  = np.zeros((Trajectory_count, Config['Input_horizon'], len(Config['Input_feature'])))
        GT_array     = np.zeros((Trajectory_count, Config['Prediction_horizon'], len(Config['Output_feature'])))
        
        Data_index   = Concat_data['DataId'].unique()
        Data_columns = Concat_data.columns
        
        Input_feature_index  = np.array([Data_columns.get_loc(feature) for feature in Config['Input_feature']])  # (Config input feature 순서대로 가져오도록 수정)
        Output_feature_index = np.array([Data_columns.get_loc(feature) for feature in Config['Output_feature']]) # (Config input feature 순서대로 가져오도록 수정)

        for index in Data_index:
            Each_data      = Concat_data[Concat_data['DataId'] == index]
            Each_unique_id = Each_data['trackId'].unique()
            
            for id in tqdm(Each_unique_id):
                Track_data     = Each_data[Each_data['trackId']==id]
                Track_data_len = len(Track_data) - Config['Input_horizon'] - Config['Prediction_horizon']
                
                if Track_data_len > 0:
                    for frame in range(0, Track_data_len, Config['Rolling_horizon']):
                        Track_input_data  = Track_data.iloc[frame : frame+Config['Input_horizon'], Input_feature_index].to_numpy()
                        Track_output_data = Track_data.iloc[frame+Config['Input_horizon'] : frame+Config['Input_horizon']+Config['Prediction_horizon'], Output_feature_index].to_numpy()
                        
                        Input_array[Array_index] = Track_input_data
                        GT_array[Array_index]    = Track_output_data
                        Array_index += 1
                        
        TENSOR    = torch.from_numpy(Input_array).type(torch.float32).to(Config['Device'])
        GT_TENSOR = torch.from_numpy(GT_array).type(torch.float32).to(Config['Device'])
        
        TENSOR_SET = TensorDataset(TENSOR, GT_TENSOR)
        LOADER     = DataLoader(TENSOR_SET, batch_size = Config['Batch_size'], shuffle = Config['Shuffle'], drop_last = True)
        
        torch.save(LOADER, Loader_dir + '/Train_loader_' + str(Config['Input_horizon']) + '_' + str(Config['Prediction_horizon'])  + '.pth')
        
    else:
        LOADER = torch.load(Loader_dir + '/Train_loader_' + str(Config['Input_horizon']) + '_' + str(Config['Prediction_horizon']) + '.pth')
                
    return LOADER

def Concat_loader_maker_for_vali(Concat_data, Trajectory_count, Config, Loader_dir):
    if len(glob.glob(Loader_dir + '/Vali_loader_' + str(Config['Input_horizon']) + '_' + str(Config['Prediction_horizon']) + '.pth')) == 0:
        print('- NO saved loader pth -')
        print('- Generating loader -')
        
        Array_index = 0
        Input_array  = np.zeros((Trajectory_count, Config['Input_horizon'], len(Config['Input_feature'])))
        GT_array     = np.zeros((Trajectory_count, Config['Prediction_horizon'], len(Config['Output_feature'])))
        
        Data_index   = Concat_data['DataId'].unique()
        Data_columns = Concat_data.columns
        
        Input_feature_index  = np.array([Data_columns.get_loc(feature) for feature in Config['Input_feature']])  # (Config input feature 순서대로 가져오도록 수정)
        Output_feature_index = np.array([Data_columns.get_loc(feature) for feature in Config['Output_feature']]) # (Config input feature 순서대로 가져오도록 수정)
        
        for index in Data_index:
            Each_data      = Concat_data[Concat_data['DataId'] == index]
            Each_unique_id = Each_data['trackId'].unique()
            
            for id in tqdm(Each_unique_id):
                Track_data     = Each_data[Each_data['trackId']==id]
                Track_data_len = len(Track_data) - Config['Input_horizon'] - Config['Prediction_horizon']
                
                if Track_data_len > 0:
                    for frame in range(0, Track_data_len, Config['Rolling_horizon']):
                        Track_input_data  = Track_data.iloc[frame : frame+Config['Input_horizon'], Input_feature_index].to_numpy()
                        Track_output_data = Track_data.iloc[frame+Config['Input_horizon'] : frame+Config['Input_horizon']+Config['Prediction_horizon'], Output_feature_index].to_numpy()
                        
                        Input_array[Array_index] = Track_input_data
                        GT_array[Array_index]    = Track_output_data
                        Array_index += 1
                        
        TENSOR    = torch.from_numpy(Input_array).type(torch.float32).to(Config['Device'])
        GT_TENSOR = torch.from_numpy(GT_array).type(torch.float32).to(Config['Device'])
        
        TENSOR_SET = TensorDataset(TENSOR, GT_TENSOR)
        LOADER     = DataLoader(TENSOR_SET, batch_size = Config['Batch_size'], shuffle = Config['Shuffle'], drop_last = True)
        
        torch.save(LOADER, Loader_dir + '/Vali_loader_' + str(Config['Input_horizon']) + '_' + str(Config['Prediction_horizon']) + '.pth')
                
    else:
        LOADER = torch.load(Loader_dir + '/Vali_loader_' + str(Config['Input_horizon']) + '_' + str(Config['Prediction_horizon']) + '.pth')
        
    return LOADER