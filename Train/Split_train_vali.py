import glob
import re
import os 
import random
import shutil
import pandas as pd
from tqdm import tqdm  

# 설정
data_dir = '../Get_data/preprocessing_maneuver'
split_ratio = (7, 3)  # train : vali 비율
random.seed(42)  # 시드 고정

Data_list = glob.glob(os.path.join(data_dir, '*.csv'))

# 각 site별 리스트 만들기
site_A_list = [Data for Data in Data_list if 0 <= int(re.findall(r'\d+', Data)[0]) <= 6]
site_B_list = [Data for Data in Data_list if 18 <= int(re.findall(r'\d+', Data)[0]) <= 29]
site_C_list = [Data for Data in Data_list if 30 <= int(re.findall(r'\d+', Data)[0]) <= 32]
site_D_list = [Data for Data in Data_list if 7 <= int(re.findall(r'\d+', Data)[0]) <= 17]

# 각 site별 디렉토리 경로
site_A_dir = './Data/site_A'
site_B_dir = './Data/site_B'
site_C_dir = './Data/site_C'
site_D_dir = './Data/site_D'

# 리스트로 정리
site_dir_list = [site_A_dir, site_B_dir, site_C_dir, site_D_dir]
site_data_lists = [site_A_list, site_B_list, site_C_list, site_D_list]

# 디렉토리 생성 및 파일 복사 및 train/vali 분리
origin_dir = "Origin"
train_dir = "TRAIN"
vali_dir = "VALI"

# tqdm을 사용하여 각 사이트별 진행 상태 표시
for site_dir, data_list in zip(site_dir_list, site_data_lists):
    origin_path = os.path.join(site_dir, origin_dir)
    train_path = os.path.join(site_dir, train_dir)
    vali_path = os.path.join(site_dir, vali_dir)

    os.makedirs(origin_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(vali_path, exist_ok=True)

    # tqdm으로 진행 상황 표시
    for file_path in tqdm(data_list, desc=f"Processing {site_dir}", unit="file"):
        filename = os.path.basename(file_path)
        dst_path = os.path.join(origin_path, filename)
        shutil.copy(file_path, dst_path)

        # CSV 읽기
        df = pd.read_csv(dst_path)
        track_ids = df['trackId'].unique().tolist()
        random.shuffle(track_ids)

        # 비율에 따라 분할
        total = len(track_ids)
        split_point = int(total * split_ratio[0] / (split_ratio[0] + split_ratio[1]))

        train_ids = set(track_ids[:split_point])
        vali_ids = set(track_ids[split_point:])

        train_df = df[df['trackId'].isin(train_ids)]
        vali_df = df[df['trackId'].isin(vali_ids)]

        # 파일명 변경 (확장자 변경)
        train_filename = filename.replace('.csv', '_train.csv')
        vali_filename = filename.replace('.csv', '_vali.csv')

        # 각각 저장
        train_df.to_csv(os.path.join(train_path, train_filename), index=False)
        vali_df.to_csv(os.path.join(vali_path, vali_filename), index=False)
