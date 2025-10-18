import cv2
import pandas as pd
import os
import glob
import re
from tqdm import tqdm
from Utils.Draw_vehicle import *

# before : 메뉴버 달기 전 차량의 좌표 및 각도 확인
# after : 메뉴버 단 후 차량의 메뉴버 확인
switch_before_maneuver = 0
switch_after_maneuver = 1

OpenData_dir = './raw'
Data_num = 18

if switch_before_maneuver:
    preprocessing_dir = './preprocessing'
    Data_list = glob.glob(preprocessing_dir + '/*.csv')
    Data_df = [Data for Data in Data_list if int(re.findall(r'\d+', Data)[0])==Data_num][0]
    Data = pd.read_csv(Data_df)

elif switch_after_maneuver:
    preprocessing_maneuver_dir = './preprocessing_maneuver'
    Data_list = glob.glob(preprocessing_maneuver_dir + '/*.csv')
    Data_df = [Data for Data in Data_list if int(re.findall(r'\d+', Data)[0])==Data_num][0]
    Data = pd.read_csv(Data_df)

# site 별 meter to pixel
if int(Data_num) >= 0 and int(Data_num) <= 6:
    M2P = 6.55 # 00~06

elif int(Data_num) >= 7 and int(Data_num) <= 17:
    M2P = 10.2 # 07~17

elif int(Data_num) >= 18 and int(Data_num) <= 29:
    M2P = 10.2 # 18~29

elif int(Data_num) >= 30 and int(Data_num) <= 32:
    M2P = 10.2 # 30~32


Unique_frame = Data['frame'].unique()
Unique_frame.sort()

target_path = "./test_image"
img_path_name = "inD_site_B_maneuver" # 여기만 바꿔서 이용
if not os.path.exists(os.path.join(target_path, img_path_name)):
    os.makedirs(os.path.join(target_path, img_path_name ))


for frame in tqdm(Unique_frame):
    
    IMG = cv2.imread(f'./{OpenData_dir}/{Data_num:02d}_background.png')

    tmp_track   = Data[Data['frame'] == frame] 

    ## [Current track id]
    Unique_track_id  = tmp_track['trackId'].unique()

    for id in Unique_track_id:
        # [Extract track data]
        Track_data = tmp_track[tmp_track['trackId']==id]
        # [Generate track points for visualization]
        Track_pts  = Draw_vehicle(Track_data, M2P)

        # [Draw vehicle]
        cv2.fillPoly(IMG, [Track_pts], color=(0, 0, 255))

        # [Put trackId]
        org = Track_pts[0]
        offset = 30          

        xCenter = "{:.2f}".format(Track_data['xCenter'].iloc[0])
        yCenter = "{:.2f}".format(Track_data['yCenter'].iloc[0])
        heading = "{:.2f}".format(Track_data['heading'].iloc[0])

        text1 = f"x: {xCenter}"
        text2 = f"y: {yCenter}"
        text3 = f"yaw: {heading}"

        if switch_before_maneuver:
            cv2.putText(IMG, text1, org=org, fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.putText(IMG, text2, org=(org[0], org[1] + offset), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.putText(IMG, text3, org=(org[0], org[1] + 2*offset), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1, color=(255, 255, 255), thickness=2)

        elif switch_after_maneuver:
            if int(Track_data['Maneuver'])==0:
                cv2.putText(IMG, str('LK'), org=Track_pts[0], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, thickness=2, color=(255, 0, 0))
            if int(Track_data['Maneuver'])==1:
                cv2.putText(IMG, str('RT'), org=Track_pts[0], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, thickness=2, color=(255, 0, 0))
            if int(Track_data['Maneuver'])==2:
                cv2.putText(IMG, str('LT'), org=Track_pts[0], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, thickness=2, color=(255, 0, 0))
            if int(Track_data['Maneuver'])==3:
                cv2.putText(IMG, str('STP'), org=Track_pts[0], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, thickness=2, color=(255, 0, 0))

    cv2.imwrite(f'./test_image/{img_path_name}/test{frame}.png', IMG)
    