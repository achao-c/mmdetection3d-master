_base_ = [
    '../_base_/models/second_hard_vfe_point_trans_rv_fusion.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_20e_fine_tune.py', '../_base_/default_runtime.py'
]
find_unused_parameters = True  # 是否查找模型中未使用的参数


load_from = '/home/yons/mmdetection3d-master/work_dirs/second_rv_fusion/epoch_40.pth'  # noqa
