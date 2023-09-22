from codes.utils.ply import read_ply
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from codes.models.utils import fps_subsample
import torch.nn.functional as F


CHOICE = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                  torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]


SEC_NAME=['_gt.ply','_partial.ply','_pred.ply']


CROP_RATIO = {
            'easy': 1/4,
            'median' :1/2,
            'hard':3/4
        }


def read_p(final_name,sec_name ):

    pc_all = []
    for n in sec_name:
        data = read_ply(f'{final_name}{n}')
        points = np.vstack((data['x'], data['y'], data['z'])).T
        pc_all.append(points)
    return pc_all[0], pc_all[1], pc_all[2]


def vis_seprate_point_cloud(points,
                        # num_points,
                        crop,
                        fixed_points=None,
                        padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    # _, n, c = xyz.shape

    # assert n == num_points
    # assert c == 3
    # if crop == num_points:
    #     return xyz, None

    # INPUT = []
    # CROP = []
    # for points in xyz:
    if isinstance(crop, list):
        num_crop = random.randint(crop[0], crop[1])
    else:
        num_crop = crop

    points = points.unsqueeze(0)

    if fixed_points is None:
        center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
    else:
        if isinstance(fixed_points, list):
            fixed_point = random.sample(fixed_points, 1)[0]
        else:
            fixed_point = fixed_points
        center = fixed_point.reshape(1, 1, 3).cuda()

    distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1),
                                    p=2,
                                    dim=-1)  # 1 1 2048

    idx = torch.argsort(distance_matrix, dim=-1,
                        descending=False)[0, 0]  # 2048

    if padding_zeros:
        input_data = points.clone()
        input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

    else:
        input_data = points.clone()[0,
                                    idx[num_crop:]].unsqueeze(0)  # 1 N 3
        
    # crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

    input_data_new=fps_subsample(input_data, 2048)[0].cpu().numpy()
    # crop_data_new=fps_subsample(crop_data, 2048)[0].cpu().numpy()

    crop_data_new = points.clone()[0, idx[:num_crop]].cpu().numpy()
    print('idx',idx.shape)
    # if isinstance(crop, list):
    #     INPUT.append(fps_subsample(input_data, 2048))
    #     CROP.append(fps_subsample(crop_data, 2048))
    # else:
    #     INPUT.append(input_data)
    #     CROP.append(crop_data)

    # input_data = torch.cat(INPUT, dim=0)  # B N 3
    # crop_data = torch.cat(CROP, dim=0)  # B M 3

    return  input_data_new, crop_data_new


folder_name='./test/'
train_folder='train_kubric_movia_Log_2023_09_09_01_16_22_old_2'
# train_folder='train_kubric_movia_Log_2023_09_13_22_01_13'
# crop_mode='easy'
crop_mode='median'
shape_name='00000000'
file_name='0000001000'
partial_id=2
final_name=f'{folder_name}/{train_folder}/kubric_movi_a_txt/{crop_mode}/outputs/{shape_name}/{file_name}_{partial_id:02}'
pc_gt, pc_partial, pc_pred = read_p(final_name, SEC_NAME )
num_crop = int(pc_gt.shape[0] * CROP_RATIO[crop_mode])
pc_pred_partial, pc_pred_npartial = vis_seprate_point_cloud(torch.from_numpy(pc_pred).cuda(), 
                                 num_crop, 
                                 fixed_points = CHOICE[partial_id])

# pc_pred_partial, pc_pred_npartial = vis_seprate_point_cloud(torch.from_numpy(pc_gt).cuda(), 
#                                  num_crop, 
#                                  fixed_points = CHOICE[partial_id])


print('pc_gt.shape',pc_gt.shape)
print('pc_partial.shape',pc_partial.shape)
print('pc_pred.shape',pc_pred.shape)
print('pc_pred_npartial.shape',pc_pred_npartial.shape)
print('pc_pred_partial.shape',pc_pred_partial.shape)

# print(pc_partial)
# print(pc_pred)


fig = plt.figure()
ax_1 = fig.add_subplot(221, projection='3d')
ax_1.scatter(pc_partial[:, 0], pc_partial[:, 1], pc_partial[:, 2])
ax_1.scatter(CHOICE[partial_id][0], CHOICE[partial_id][1], CHOICE[partial_id][2])

ax_2 = fig.add_subplot(222, projection='3d')
ax_2.scatter(pc_pred[:, 0], pc_pred[:, 1], pc_pred[:, 2])

ax_3 = fig.add_subplot(223, projection='3d')
# ax_3.scatter(pc_partial[:, 0], pc_partial[:, 1], pc_partial[:, 2])
ax_3.scatter(pc_pred_partial[:, 0], pc_pred_partial[:, 1], pc_pred_partial[:, 2])
ax_3.scatter(pc_pred_npartial[:, 0], pc_pred_npartial[:, 1], pc_pred_npartial[:, 2])
# ax_3.scatter(pc_pred[:, 0], pc_pred[:, 1], pc_pred[:, 2])

ax_4 = fig.add_subplot(224, projection='3d')
ax_4.scatter(pc_gt[:, 0], pc_gt[:, 1], pc_gt[:, 2])


plt.show()