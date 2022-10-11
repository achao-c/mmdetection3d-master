import torch
dict = torch.load('/home/yons/mmdetection3d-master/work_dirs/second_hard_vfe_points_trans/epoch_40.pth')
print(dict['state_dict'].keys())