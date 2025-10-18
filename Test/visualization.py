import natsort
import pandas as pd
import glob
import cv2
import os
from tqdm import tqdm
from Models.Dual_transformer import *
from Utils.Base import *
from Utils.Draw_vehicle import *
from Utils.Make_tensor import *
from Utils.Average_deviation import *
from PIL import Image
import imageio

## setting
Data_num = 18
Data_name = '../Get_data/preprocessing_maneuver/18_tracks_del_vru_static_maneuver.csv'
log_dir       = r'Z:\정선암\졸업\Code\inD_vehicle_B\log\Performance\Dual_transformer_Config_TEST_25_75_2024-10-16-21-25'
img_path_name = "inD_site_B_visualization" 
## setting

OpenData_dir = '../Get_data/raw'
Data = pd.read_csv(Data_name) 
# site 별 meter to pixel
if int(Data_num) >= 0 and int(Data_num) <= 6:
    M2P = 6.55 # 00~06

elif int(Data_num) >= 7 and int(Data_num) <= 17:
    M2P = 10.2 # 07~17

elif int(Data_num) >= 18 and int(Data_num) <= 29:
    M2P = 10.2 # 18~29

elif int(Data_num) >= 30 and int(Data_num) <= 32:
    M2P = 10.2 # 30~32


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

# Model
TEST_MODEL = Dual_transformer(Config_dict).to(Config_dict['Device'])
TEST_MODEL.load_state_dict(torch.load(latest_file_path, map_location='cuda:0'))
TEST_MODEL.eval()
TEST_MODEL = TEST_MODEL.cuda()

# visualization
INPUT_FRAME  = Config_dict['Input_horizon']
OUTPUT_FRAME = Config_dict['Prediction_horizon']

Unique_frame = Data['frame'].unique()
Unique_frame.sort()

target_path = "./test_image"
if not os.path.exists(os.path.join(target_path, img_path_name)):
    os.makedirs(os.path.join(target_path, img_path_name ))

for frame in tqdm(Unique_frame):

    IMG = cv2.imread(f'./{OpenData_dir}/{Data_num:02d}_background.png')

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
            # [Make tensor]
            Tensor_data = Make_tensor(Track_data, Config_dict)
            # [Generate track points for visualization]
            Track_pts  = Draw_vehicle(Track_data, M2P)
            # [Predict]
            with torch.no_grad():
                Trajectory, *_, = TEST_MODEL(Tensor_data.unsqueeze(0), Tensor_data.unsqueeze(0), Tensor_data.unsqueeze(0)) 
                Trajectory = Trajectory[:, :OUTPUT_FRAME, :2]

                Trajectory[:, :, 0] =  reverse_standard_scale(Trajectory[:, :, 0],  Config_dict['mean'][0], Config_dict['std'][0])
                Trajectory[:, :, 1] =  reverse_standard_scale(Trajectory[:, :, 1], Config_dict['mean'][1], Config_dict['std'][1])

            Trajectory_img = Trajectory.squeeze(0).detach().cpu() * M2P
            Trajectory_img = np.array(Trajectory_img, np.int32)
            Trajectory_img[:,1] = -Trajectory_img[:,1]

            GT_traj_img = GT_traj * M2P
            GT_traj_img[:, 1] = -GT_traj_img[:, 1]
            GT_TRAJ = np.array(GT_traj_img, np.int32)

            # [Draw trajectories]
            for i in Trajectory_img:
                cv2.circle(IMG, i, 1, (255, 0, 255), 2)

            # [Draw trajectories]
            for i in GT_TRAJ:
                cv2.circle(IMG, i, 1, (255, 0, 0), 2)
                    
            # [Draw vehicle]
            cv2.fillPoly(IMG, [Track_pts], color=(0, 0, 255))

            # [Average Deviation Lateral]
            GT_trajectory_x = GT_traj[:,0]
            GT_trajectory_y = GT_traj[:,1]

            Pred_trajectory_x = Trajectory[0, :, 0].detach().cpu().numpy()
            Pred_trajectory_y = Trajectory[0, :, 1].detach().cpu().numpy()

            Average_Deviation_lateral = Average_deviation(GT_trajectory_x, GT_trajectory_y, Pred_trajectory_x, Pred_trajectory_y)

            cv2.putText(IMG, f'AD: {Average_Deviation_lateral:.2f}', 
            org=Track_pts[0], 
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=0.8, 
            color=(0, 255, 0))

        else:
            # [Generate track points for visualization]
            Track_pts  = Draw_vehicle(Track_data, M2P)
            # [Draw vehicle]
            cv2.fillPoly(IMG, [Track_pts], color=(0, 0, 255))

        cv2.putText(IMG, str(id), org=Track_pts[0]+20, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(150, 150, 150))


    cv2.imwrite(f'./test_image/{img_path_name}/test{frame}.png', IMG)

# gif 
img_path = [f"./test_image/" + img_path_name + "/" + i for i in natsort.natsorted(os.listdir("./test_image/" + img_path_name))]
imgs = [ Image.open(i) for i in img_path]

save_path= './gif/'

imageio.mimsave(save_path +img_path_name+ '.gif', imgs, duration=19)