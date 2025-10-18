# [IMPORT LIBRARY]
import time
from Utils.Base import *
from Utils.Loader_v5 import *
from Utils.Train_dual_transformer import *
from Models.Dual_transformer import *

###################################################################################################

# [LOAD CONFIGURATION]
site = "site_B"
TRAIN_DATA_DIR     = f'./Data/{site}/TRAIN/**.csv'
VALI_DATA_DIR      = f'./Data/{site}/VALI/**.csv'

LOADER_CONFIG_DIR  = './Config/Loader_config.json'
LOADER_CONFIG      = Read_config(LOADER_CONFIG_DIR)

NETWORK_CONFIG_DIR = './Config/Network_config.json'
NETWORK_CONFIG     = Read_config(NETWORK_CONFIG_DIR)
NETWORK_CONFIG['Input_horizon']      = LOADER_CONFIG['Input_horizon']
NETWORK_CONFIG['Prediction_horizon'] = LOADER_CONFIG['Prediction_horizon']
###################################################################################################
# [PREPROCESSING]
Train_track_count, Train_trajectory_count = Count_trajectory(TRAIN_DATA_DIR, LOADER_CONFIG)
Vali_track_count, Vali_trajectory_count   = Count_trajectory_for_vali(VALI_DATA_DIR, LOADER_CONFIG)

Train_scaled_concat_data, Concat_mean, Concat_std = Concat_frame_maker(TRAIN_DATA_DIR, LOADER_CONFIG)
Vali_scaeld_concat_data = Concat_frame_maker_for_vali(VALI_DATA_DIR, LOADER_CONFIG, Concat_mean, Concat_std)
###################################################################################################
# [GENERATE LOADER]
LOADER_DIR = f'./Loader/{site}'
os.makedirs(LOADER_DIR, exist_ok=True)

TRAIN_LOADER = Concat_loader_maker(Train_scaled_concat_data, Train_trajectory_count, LOADER_CONFIG, LOADER_DIR)
VALI_LOADER  = Concat_loader_maker_for_vali(Vali_scaeld_concat_data, Vali_trajectory_count, LOADER_CONFIG, LOADER_DIR)
###################################################################################################
# [SET NETWORK]
MODEL = Dual_transformer(NETWORK_CONFIG).to(NETWORK_CONFIG['Device'])
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.001)
SCHEDULER = torch.optim.lr_scheduler.OneCycleLR(OPTIMIZER, max_lr = 0.01, steps_per_epoch = len(TRAIN_LOADER), epochs=NETWORK_CONFIG['Epochs'])
###################################################################################################
# [Start Train]
MEANSTD_CONFIG_DIR = './Config/Meanstd_config.json'
MEANSTD_CONFIG = Read_config(MEANSTD_CONFIG_DIR)

start_time = time.time()
Model_train(MODEL, TRAIN_LOADER, VALI_LOADER, OPTIMIZER, SCHEDULER, NETWORK_CONFIG, LOADER_CONFIG, Concat_mean, Concat_std, MEANSTD_CONFIG)
end_time = time.time()

# 코드 실행 시간 계산
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print("학습 시간:", minutes, "분", seconds, "초")