import pandas as pd
import numpy as np
import glob
import os
import natsort

Data_tracks_list = natsort.natsorted(glob.glob(f'./raw/**tracks.csv'))
Data_tracksMeta_list = natsort.natsorted(glob.glob(f'./raw/**tracksMeta.csv'))

# save path
folder_path = 'preprocessing'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# del vra & static
for data1, data2 in zip(Data_tracks_list, Data_tracksMeta_list):

    Name = data1.split('\\')[-1].replace('.csv', '_del_vru_static.csv') 
    
    Data_tracks  = pd.read_csv(data1)
    Data_tracksMeta = pd.read_csv(data2)
    
    Extract_vru = Data_tracksMeta[(Data_tracksMeta['width'] == 0) | (Data_tracksMeta['length'] == 0)]['trackId'].unique()
    
    Data_tracks = Data_tracks[~Data_tracks['trackId'].isin(Extract_vru)].reset_index(drop=True)
    
    Unique_track_id = Data_tracks['trackId'].unique()
    
    for track_id in Unique_track_id:
        Track_data = Data_tracks[Data_tracks['trackId'] == track_id]
        
        if np.mean(Track_data['xVelocity']) == 0.0 and np.mean(Track_data['yVelocity']) == 0.0:
            print('del:', track_id)
            Data_tracks = Data_tracks.drop(Track_data.index)

    Data_tracks.to_csv(f'./{folder_path}/{Name}', index=False)
    
    