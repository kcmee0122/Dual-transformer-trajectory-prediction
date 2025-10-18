import pandas as pd
import numpy as np
import glob
import re
import os
from tqdm import tqdm
from Utils.Draw_vehicle_pts import *

preprocessing_dir = './preprocessing'
Data_list = glob.glob(preprocessing_dir + '/*.csv')
Data_list = [Data for Data in Data_list if int(re.findall(r'\d+', Data)[0])>=7 and int(re.findall(r'\d+', Data)[0])<=17] # site D 

folder_path = 'preprocessing_maneuver'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for data_dir in tqdm(Data_list):
    Data = pd.read_csv(data_dir)
    Name = data_dir.split('\\')[-1].replace('.csv', '_maneuver.csv')
    Filter_dict = {}

    TOTAL_ID = []

    RT_21, RT_43, RT_32, RT_14 = [], [], [], []

    LT_12, LT_34, LT_23, LT_41 = [], [], [], []

    # [First filtering]
    for id in Data['trackId'].unique():
        Track_data = Data[Data['trackId'] == id]
        
        Init_heading = Track_data['heading'].iloc[0]
        End_heading  = Track_data['heading'].iloc[-1]
        ################################################################
        ## [2 -> 1]
        if (200 <= Init_heading <= 230) & (120 <= End_heading <= 140):
            # Filter_dict[id] = 'RT_21'
            RT_21.append(id)
            TOTAL_ID.append(id)
            
        ## [4 -> 3]
        elif (30 <= Init_heading <= 50) & (300 <= End_heading <= 320):
            # Filter_dict[id] = 'RT_43'
            RT_43.append(id)
            TOTAL_ID.append(id)
            
        ## [3 -> 2]
        elif (120 <= Init_heading <= 140) & (40 <= End_heading <= 90):
            # Filter_dict[id] = 'RT_32'
            RT_32.append(id)
            TOTAL_ID.append(id)
        
        ## [1 -> 4]
        elif (100 <= Init_heading <= 120) & (5 <= End_heading <= 20):
            # Filter_dict[id] = 'RT_14'
            RT_14.append(id)
            TOTAL_ID.append(id)
        ################################################################
        ## [1 -> 2]
        if (300 <= Init_heading <= 320) & (40 <= End_heading <= 60):
            # Filter_dict[id] = 'LT_12'
            LT_12.append(id)
            TOTAL_ID.append(id)
            
        ## [3 -> 4]    
        elif (260 <= Init_heading <= 330) & (5 <= End_heading <= 100):
            # Filter_dict[id] = 'LT_34'
            LT_34.append(id)
            TOTAL_ID.append(id)
            
        ## [2 -> 3]
        elif (210 <= Init_heading <= 240) & (300 <= End_heading <= 320):
            # Filter_dict[id] = 'LT_23'
            LT_23.append(id)
            TOTAL_ID.append(id)
            
        ## [4 -> 1]
        elif (30 <= Init_heading <= 50) & (120 <= End_heading <= 140):
            # Filter_dict[id] = 'LT_41'
            LT_41.append(id)
            TOTAL_ID.append(id)
        ################################################################
            
    # [Second filtering]
    ###############################################################################################################################################################

    line1_a = -1.1031
    line1_b = 35.4614

    line2_a = -1.1031
    line2_b = 21.896

    line3_a = 0.90656
    line3_b = -72.8604

    line4_a = 0.90656
    line4_b = -91.908

    ###############################################################################################################################################################
    # RT : 1
    Temp_RT_21 = Data[Data['trackId'].isin(RT_21)]
    RT_21_index = []

    for index, row in Temp_RT_21.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y <= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y >= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])


        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y >= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y <= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            RT_21_index.append(index)

    Data.loc[RT_21_index, 'Maneuver'] = 1
    ################################################################
    Temp_RT_43 = Data[Data['trackId'].isin(RT_43)]
    RT_43_index = []

    for index, row in Temp_RT_43.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y >= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y <= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])


        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y <= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y >= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            RT_43_index.append(index)
    Data.loc[RT_43_index, 'Maneuver'] = 1
    ################################################################
    Temp_RT_32 = Data[Data['trackId'].isin(RT_32)]
    RT_32_index = []

    for index, row in Temp_RT_32.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y >= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y >= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])

        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y <= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y <= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            RT_32_index.append(index)
    Data.loc[RT_32_index, 'Maneuver'] = 1
    ################################################################
    Temp_RT_14 = Data[Data['trackId'].isin(RT_14)]
    RT_14_index = []

    for index, row in Temp_RT_14.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y <= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y <= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])

        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y >= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y >= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            RT_14_index.append(index)
    Data.loc[RT_14_index, 'Maneuver'] = 1
    ###############################################################################################################################################################
    # LT : 2
    Temp_LT_12 = Data[Data['trackId'].isin(LT_12)]
    LT_12_index = []

    for index, row in Temp_LT_12.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y <= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y >= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])

        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y >= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y <= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            LT_12_index.append(index)

    Data.loc[LT_12_index, 'Maneuver'] = 2
    ################################################################
    Temp_LT_34 = Data[Data['trackId'].isin(LT_34)]
    LT_34_index = []

    for index, row in Temp_LT_34.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y >= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y <= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])

        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y <= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y >= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            LT_34_index.append(index)
    Data.loc[LT_34_index, 'Maneuver'] = 2
    ################################################################
    Temp_LT_23 = Data[Data['trackId'].isin(LT_23)]
    LT_23_index = []

    for index, row in Temp_LT_23.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y >= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y >= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])

        # line 1
        y = line1_a * Track_pts[:,0] + line1_b
        result_1 = y <= Track_pts[:,1]

        # line 4
        y = line4_a * Track_pts[:,0] + line4_b
        result_2 = y <= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            LT_23_index.append(index)
    Data.loc[LT_23_index, 'Maneuver'] = 2
    ################################################################
    Temp_LT_41 = Data[Data['trackId'].isin(LT_41)]
    LT_41_index = []

    for index, row in Temp_LT_41.iterrows():
        Track_pts = Draw_vehicle_pts(row)

        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y <= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y <= Track_pts[:,1]

        LK_result_1 = np.all([result_1, result_2])

        # line 2
        y = line2_a * Track_pts[:,0] + line2_b
        result_1 = y >= Track_pts[:,1]

        # line 3
        y = line3_a * Track_pts[:,0] + line3_b
        result_2 = y >= Track_pts[:,1]

        LK_result_2 = np.all([result_1, result_2])

        if (LK_result_1 == False) and (LK_result_2 == False):
            LT_41_index.append(index)
    Data.loc[LT_41_index, 'Maneuver'] = 2
    ###############################################################################################################################################################

    Data.loc[(abs(Data['xVelocity']) < 0.05) & (abs(Data['yVelocity']) < 0.05), 'Maneuver'] = 3
    Data.fillna(0).to_csv(folder_path + '/' +  Name, index=False)
