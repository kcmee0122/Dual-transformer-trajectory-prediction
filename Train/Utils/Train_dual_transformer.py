import os
import time
import json
import torch
import numpy as np
import torch.nn as nn

from Utils.Base import *
from Utils.Count_params import *

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, recall_score, classification_report

def Model_train(Model, Train_loader, Vali_loader, Optimizer, Scheduler, Network_Config, Loader_config, Tensor_mean, Tensor_std, MEANSTD_CONFIG):
    ####################################################################################
    print('- Save Config dictionary -')
    dict_name = Model._get_name() + f'_Config_' + str(Network_Config['Type']) + '_' + str(Loader_config['Input_horizon']) + '_' + str(Loader_config['Prediction_horizon']) + '_' + str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))
    dict_name_Loader = Model._get_name() + f'_Config_' + str(Network_Config['Type']) + '_' + str(Loader_config['Input_horizon']) + '_' + str(Loader_config['Prediction_horizon']) + '_Loader_' + str(time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())))
    dict_name_meanstd = 'Meanstd_config'
    if os.path.exists('./log') == False:
        os.makedirs('./log')
        
    if os.path.exists('./log/' + dict_name) == False:
        os.makedirs('./log/' + dict_name)
        
    if os.path.exists('./log/' + dict_name + '/pt') == False:
        os.makedirs('./log/' + dict_name + '/pt')
    
    with open(f"./log/{dict_name}/{dict_name}.json", 'w') as file:
        file.write(json.dumps(Network_Config, indent=4)) # use `json.loads` to do the reverse

    with open(f"./log/{dict_name}/{dict_name_Loader}.json", 'w') as file:
        file.write(json.dumps(Loader_config, indent=4)) # use `json.loads` to do the reverse

    with open(f"./log/{dict_name}/{dict_name_meanstd}.json", 'w') as file:
        file.write(json.dumps(MEANSTD_CONFIG, indent=4)) # use `json.loads` to do the reverse
    ####################################################################################
    # [Start]
    print('Train start:')
    # [Set loss variables for saving .pt]
    Best_trajectory_loss_x   = np.inf
    Best_trajectory_loss_y   = np.inf
    Best_classification_loss = np.inf
    
    # [Set criterion]
    Criterion_for_mse = nn.MSELoss(reduction='mean')
    Criterion_for_ce  = nn.CrossEntropyLoss()
    
    # [Set variables for reverse scaling]
    X_mean, X_std = Tensor_mean[0], Tensor_std[0]
    Y_mean, Y_std = Tensor_mean[1], Tensor_std[1]
    
    # [Set epoch and model]
    for epoch in range(Network_Config['Epochs']):
        Model.train()
        
        # [Set check point]
        if (epoch != 0) and (epoch % 100 == 0):
            print('Check point epoch:', epoch)
        
        ###################################
        # [Set loss variables]
        ## [MSE]
        Total_train_trajectory_mse_x = 0
        Total_train_trajectory_mse_y = 0
        
        ## [ADE]
        Total_train_trajectory_ade = 0
        
        ## [FDE]
        Total_train_trajectory_fde = 0
        
        ## [CE]
        Total_train_maneuver_loss = 0

        ###################################
        
        ###################################
        # [Train]
        for index, (Train_input, Train_gt) in enumerate(tqdm(Train_loader)):
            # [Grad reset]
            Optimizer.zero_grad()
            
            # [Inference]
            ## [Batch, Seq_len, Output_features]
            Train_prediction, Train_maneuver_prediction = Model(Train_input, Train_input, Train_input)
            
            # [Calculate loss]
            Train_loss    = Criterion_for_mse(Train_prediction, Train_gt[:, :, :2])
            Train_loss_ce = Criterion_for_ce(Train_maneuver_prediction[:, -1, :], Train_gt[:, 0:1, 2:].view(-1).type(torch.LongTensor).to('cuda:0'))

            
            # [Backprob]
            Total_loss = Train_loss_ce + Train_loss 
            Total_loss.backward()

            
            # [Clipping]
            nn.utils.clip_grad_norm_(Model.parameters(), max_norm = 1)
            
            
            # [Optimizer update]
            Optimizer.step()
            
            # [Scheduler updata]
            ## [However, if you are using a scheduler like CosineAnnealingLR or CyclicLR, you might want to adjust the learning rate after each mini-batch]
            Scheduler.step()
            
            ###################################
            # [Calculate KPI]
            with torch.no_grad():
                Train_trajectory_x = Tensor_reverse_standard_scale(Train_prediction[:, :, 0].detach(), X_mean, X_std)
                Train_trajectory_y = Tensor_reverse_standard_scale(Train_prediction[:, :, 1].detach(), Y_mean, Y_std)
                
                Train_trajectory_gt_x = Tensor_reverse_standard_scale(Train_gt[:, :, 0].detach(), X_mean, X_std)
                Train_trajectory_gt_y = Tensor_reverse_standard_scale(Train_gt[:, :, 1].detach(), Y_mean, Y_std)
                
                Train_trajectory_loss_x = Criterion_for_mse(Train_trajectory_x, Train_trajectory_gt_x)
                Train_trajectory_loss_y = Criterion_for_mse(Train_trajectory_y, Train_trajectory_gt_y) 
                
                Total_train_trajectory_mse_x += Train_trajectory_loss_x
                Total_train_trajectory_mse_y += Train_trajectory_loss_y

                Total_train_trajectory_fde += torch.sum(torch.sqrt((Train_trajectory_x[:, -1:] - Train_trajectory_gt_x[:, -1:])**2 + (Train_trajectory_y[:, -1:] - Train_trajectory_gt_y[:, -1:])**2))
                Total_train_trajectory_ade += torch.sum(torch.sqrt((Train_trajectory_x - Train_trajectory_gt_x)**2 + (Train_trajectory_y - Train_trajectory_gt_y)**2))
                
                Total_train_maneuver_loss += Train_loss_ce.detach()
            ###################################
                
        ###################################
        # [Calculate train MSE]
        Total_train_trajectory_mse_x = Total_train_trajectory_mse_x / len(Train_loader)
        Total_train_trajectory_mse_y = Total_train_trajectory_mse_y / len(Train_loader)
        
        Total_train_trajectory_rmse_x = torch.sqrt(Total_train_trajectory_mse_x)
        Total_train_trajectory_rmse_y = torch.sqrt(Total_train_trajectory_mse_y)

        Total_train_trajectory_fde = Total_train_trajectory_fde / (Loader_config['Batch_size'] * len(Train_loader))
        Total_train_trajectory_ade = Total_train_trajectory_ade / (Loader_config['Batch_size'] * len(Train_loader) * Loader_config['Prediction_horizon'])
        
        Total_train_maneuver_loss = Total_train_maneuver_loss / len(Train_loader)
        ###################################
        # [Validation]
        Model.eval()

        with torch.no_grad():
            # [Set loss variables]
            ## [MSE]
            Total_vali_trajectory_mse_x = 0
            Total_vali_trajectory_mse_y = 0
            
            ## [ADE]
            Total_vali_trajectory_ade = 0
            
            ## [FDE]
            Total_vali_trajectory_fde = 0
            
            ## [CE]
            Total_vali_maneuver_loss  = 0
            Total_vali_maneuver_loss2 = 0
            Total_vali_maneuver_loss3 = 0
            
            Maneuver_gt   = []
            Maneuver_pred = []
            
            for Vali_input, Vali_gt in tqdm(Vali_loader):
                # [Inference]
                ## [Batch, Seq_len, Output_features]
                Vali_prediction, Vali_maneuver_prediction = Model(Vali_input, Vali_input, Vali_input)
                
                # [Calculate KPI]
                Vali_trajectory_x = Tensor_reverse_standard_scale(Vali_prediction[:, :, 0].detach(), X_mean, X_std)
                Vali_trajectory_y = Tensor_reverse_standard_scale(Vali_prediction[:, :, 1].detach(), Y_mean, Y_std)
                
                Vali_trajectory_gt_x = Tensor_reverse_standard_scale(Vali_gt[:, :, 0].detach(), X_mean, X_std)
                Vali_trajectory_gt_y = Tensor_reverse_standard_scale(Vali_gt[:, :, 1].detach(), Y_mean, Y_std)
                
                Vali_trajectory_loss_x = Criterion_for_mse(Vali_trajectory_x, Vali_trajectory_gt_x)
                Vali_trajectory_loss_y = Criterion_for_mse(Vali_trajectory_y, Vali_trajectory_gt_y) 
                Vali_maneuver_loss     = Criterion_for_ce(Vali_maneuver_prediction[:, -1, :], Vali_gt[:, 0:1, 2:].view(-1).type(torch.LongTensor).to('cuda:0'))
                
                Total_vali_trajectory_mse_x += Vali_trajectory_loss_x
                Total_vali_trajectory_mse_y += Vali_trajectory_loss_y
                Total_vali_maneuver_loss    += Vali_maneuver_loss

                Total_vali_trajectory_fde += torch.sum(torch.sqrt((Vali_trajectory_x[:, -1:] - Vali_trajectory_gt_x[:, -1:])**2 + (Vali_trajectory_y[:, -1:] - Vali_trajectory_gt_y[:, -1:])**2))
                Total_vali_trajectory_ade += torch.sum(torch.sqrt((Vali_trajectory_x - Vali_trajectory_gt_x)**2 + (Vali_trajectory_y - Vali_trajectory_gt_y)**2))
                
                _, predicted = torch.max(Vali_maneuver_prediction[:, -1, :], -1)
                Total_vali_maneuver_loss2 += Vali_gt.size(0)
                Total_vali_maneuver_loss3 += (predicted == Vali_gt[:, 0:1, 2:].view(-1).type(torch.LongTensor).to('cuda:0')).sum().item()
                
                Maneuver_gt.extend(Vali_gt[:, 0:1, 2:].view(-1).type(torch.LongTensor).to('cuda:0').detach().cpu().numpy().tolist())
                Maneuver_pred.extend(predicted.detach().cpu().numpy().tolist())
                        
            ###################################
            # [Calculate train MSE]
            Total_vali_trajectory_mse_x = Total_vali_trajectory_mse_x / len(Vali_loader)
            Total_vali_trajectory_mse_y = Total_vali_trajectory_mse_y / len(Vali_loader)
            
            Total_vali_trajectory_rmse_x = torch.sqrt(Total_vali_trajectory_mse_x)
            Total_vali_trajectory_rmse_y = torch.sqrt(Total_vali_trajectory_mse_y)

            Total_vali_trajectory_fde = Total_vali_trajectory_fde /  (Loader_config['Batch_size'] * len(Vali_loader))
            Total_vali_trajectory_ade = Total_vali_trajectory_ade / (Loader_config['Batch_size'] * len(Vali_loader) * Loader_config['Prediction_horizon'])
            
            Total_vali_maneuver_loss = Total_vali_maneuver_loss / len(Vali_loader)

            ###################################

        if ((Best_trajectory_loss_x >= Total_vali_trajectory_mse_x) and (Best_trajectory_loss_y >= Total_vali_trajectory_mse_y)):
                Best_trajectory_loss_x = Total_vali_trajectory_mse_x
                Best_trajectory_loss_y = Total_vali_trajectory_mse_y
                Best_classification_loss = Total_vali_maneuver_loss

                print('Train RMSE_X:', Total_train_trajectory_rmse_x)
                print('Train RMSE_Y:', Total_train_trajectory_rmse_y)
                print('Train ADE:', Total_train_trajectory_ade)
                print('Train FDE:', Total_train_trajectory_fde)

                print('Vali RMSE_X:', Total_vali_trajectory_rmse_x)
                print('Vali RMSE_Y:', Total_vali_trajectory_rmse_y)
                print('Vali ADE:', Total_vali_trajectory_ade)
                print('Vali FDE:', Total_vali_trajectory_fde)
                print('Vali report:')
                report = classification_report(Maneuver_gt, Maneuver_pred)
                print(report)
                print('epoch -> ', epoch, '/', Network_Config['Epochs'])
                
                if epoch == 0:
                    f = open(f'./log/{dict_name}/' + time.strftime('%Y-%m-%d-%H', time.localtime(time.time())) + '.txt', 'w')
                    f.write('Trainable params: ' + str(Count_parameters(Model)) + '\n')
                    f.write('Epoch' + str(epoch) + '\n')
                    f.write('Vali MSE: ' + str(Total_vali_trajectory_mse_x) + ' ' + str(Total_vali_trajectory_mse_y) + '\n')
                    f.write('Vali RMSE: ' + str(Total_vali_trajectory_rmse_x) + ' ' + str(Total_vali_trajectory_rmse_y) + '\n')
                    f.write('Vali ADE: ' + str(Total_vali_trajectory_ade) + '\n')
                    f.write('Vali FDE: ' + str(Total_vali_trajectory_fde) + '\n')
                    f.write('Vali report:' + '\n')
                    f.write(report + '\n')
                    f.close()
                    
                else:
                    f = open(f'./log/{dict_name}/' + time.strftime('%Y-%m-%d-%H', time.localtime(time.time())) + '.txt', 'a')
                    f.write('Epoch' + str(epoch) + '\n')
                    f.write('Vali MSE: ' + str(Total_vali_trajectory_mse_x) + ' ' + str(Total_vali_trajectory_mse_y) + '\n')
                    f.write('Vali RMSE: ' + str(Total_vali_trajectory_rmse_x) + ' ' + str(Total_vali_trajectory_rmse_y) + '\n')
                    f.write('Vali ADE: ' + str(Total_vali_trajectory_ade) + '\n')
                    f.write('Vali FDE: ' + str(Total_vali_trajectory_fde) + '\n')
                    f.write('Vali report:' +'\n')
                    f.write(report + '\n')
                    f.close()
                                        
                Save_name = f'./log/{dict_name}/pt/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time())) + str('_Epoch_') + str(epoch) + '_' + str(Loader_config['Input_horizon']) + '_' + str(Loader_config['Prediction_horizon']) + '.pt'
                torch.save(Model.state_dict(), Save_name)
            