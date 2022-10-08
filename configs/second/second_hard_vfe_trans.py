_base_ = [
    '../_base_/models/second_hard_vfe_trans.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

find_unused_parameters = True  # 是否查找模型中未使用的参数