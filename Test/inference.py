# [IMPORT LIBRARY]
import natsort
import glob
import os
import time
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from Utils.Base import *
from Utils.Make_tensor import *
from Utils.Average_deviation import *
from Models.Dual_transformer import *

start_time = time.time()

## setting
Data_dir = r'Z:\정선암\졸업\Code\inD_vehicle_B\Data\VALI'
log_dir       = r'Z:\정선암\졸업\Code\inD_vehicle_B\log\Performance\Dual_transformer_Config_TEST_25_75_2024-10-16-21-25'
## setting

# JSON 파일 읽어오기
Config_json      = f'{log_dir}/**.json'

json_list = natsort.natsorted(glob.glob(Config_json))

Config_dict = {}
for json in json_list:
    tmp_Config_dict = Read_config(json)
    Config_dict = {**Config_dict, **tmp_Config_dict}

if Config_dict:
    print("The dictionary has content.")
else:
    print("The dictionary is empty.")

weigth_dir      = f'{log_dir}/pt/**.pt'
weight_list = natsort.natsorted(glob.glob(weigth_dir))

# 최근 수정된 파일의 경로와 수정 시간 초기화
latest_file_path = None
latest_mtime = 0

# load latest weight file 
for file_path in weight_list:
    mtime = os.path.getmtime(file_path)
    if mtime > latest_mtime:
        latest_mtime = mtime
        latest_file_path = file_path

TEST_MODEL = Dual_transformer(Config_dict).to(Config_dict['Device'])
TEST_MODEL.load_state_dict(torch.load(latest_file_path, map_location='cuda:0'))
TEST_MODEL.eval()
TEST_MODEL = TEST_MODEL.cuda()

# 평가
Criterion_for_mse = nn.MSELoss(reduction='mean')

INPUT_FRAME  = Config_dict['Input_horizon']
OUTPUT_FRAME = Config_dict['Prediction_horizon']

Data_csv =  f'{Data_dir}/**.csv'
Data_list = natsort.natsorted(glob.glob(Data_csv))

mse_x_all = 0
mse_y_all = 0
ade_all = 0
fde_all = 0
AD_all = 0
traj_num = 0

for Data in Data_list:
    Data = pd.read_csv(Data)
    
    Unique_frame = Data['frame'].unique()
    Unique_frame.sort()

    for frame in tqdm(Unique_frame):
        ## [Current data]
        Original_input   = Data[Data['frame'] == frame] 
        ## [Historical data]
        Assumption_input = Data[((frame >= Data['frame']) & (Data['frame'] > frame-INPUT_FRAME))]
        ## [Current track id]
        Unique_track_id  = Original_input['trackId'].unique()

        GT = Data[(frame + 1 <= Data['frame']) & (Data['frame'] <= frame + OUTPUT_FRAME)]

        for id in Unique_track_id:

            # [Extract track data]
            Track_data = Assumption_input[Assumption_input['trackId']==id]
            GT_data = GT[GT['trackId']==id]
            GT_traj = np.array(GT_data.loc[:, ['xCenter', 'yCenter']])

            if len(Track_data) == INPUT_FRAME and len(GT_data) == OUTPUT_FRAME:
                # [Count traj]
                traj_num+=1
                # [Make tensor]
                Tensor_data = Make_tensor(Track_data, Config_dict)

                # [Predict]
                with torch.no_grad():
                    Trajectory, *_, = TEST_MODEL(Tensor_data.unsqueeze(0), Tensor_data.unsqueeze(0), Tensor_data.unsqueeze(0)) 
                    Trajectory = Trajectory[:, :OUTPUT_FRAME, :2]
                    # [Reverse scale]
                    Trajectory[:, :, 0] =  reverse_standard_scale(Trajectory[:, :, 0],  Config_dict['mean'][0], Config_dict['std'][0])
                    Trajectory[:, :, 1] =  reverse_standard_scale(Trajectory[:, :, 1], Config_dict['mean'][1], Config_dict['std'][1])
                    pred = Trajectory.squeeze(0)

                # [Performance]
                pred_x = pred[:, 0]
                pred_y = pred[:, 1]

                gt_x = torch.tensor(GT_traj[:, 0], device= Config_dict['Device'])
                gt_y = torch.tensor(GT_traj[:, 1], device= Config_dict['Device'])

                mse_x = Criterion_for_mse(pred_x, gt_x)
                mse_y = Criterion_for_mse(pred_y, gt_y)

                diff_x = pred_x - gt_x
                diff_y = pred_y - gt_y

                l2_dist = torch.sqrt(diff_x ** 2 + diff_y ** 2)

                ade = l2_dist.mean()
                fde = l2_dist[-1]

                Average_Deviation_lateral = Average_deviation(gt_x, gt_y, pred_x, pred_y)

                mse_x_all+= mse_x
                mse_y_all+= mse_y
                ade_all+= ade
                fde_all+= fde
                AD_all+= Average_Deviation_lateral

mse_x_total = mse_x_all / traj_num
mse_y_total = mse_y_all / traj_num
rmse_x = torch.sqrt(mse_x_total)
rmse_y = torch.sqrt(mse_y_total)

ade_total = ade_all / traj_num
fde_total = fde_all / traj_num
AD_total = AD_all / traj_num

print(f"RMSE_x: {rmse_x.item():.4f}")
print(f"RMSE_y: {rmse_y.item():.4f}")
print(f"ADE: {ade_total.item():.4f}")
print(f"FDE: {fde_total.item():.4f}")
print(f"Average_deviation: {AD_total.item():.4f}")

end_time = time.time()

# 코드 실행 시간 계산
execution_time = end_time - start_time
minutes = int(execution_time // 60)
seconds = int(execution_time % 60)
print("소요 시간:", minutes, "분", seconds, "초")
