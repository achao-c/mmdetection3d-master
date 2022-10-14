# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector

import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
@DETECTORS.register_module()
class MVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN, self).__init__(**kwargs)



class fusion_Block(nn.Module):
    def __init__(self, l_c, p_c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(p_c, p_c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.fusion_conv = ConvModule(
            l_c,
            l_c,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            act_cfg=dict(type='ReLU'),
            inplace=False)
    def forward(self, lfeat, pfeat):
        pfeat = self.att(pfeat) * pfeat
        lfeat_add = pfeat + lfeat
        return self.fusion_conv(lfeat_add)

@DETECTORS.register_module()
class Lp_fusion(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(Lp_fusion, self).__init__(**kwargs)
        self.fusion_block = fusion_Block(256, 256)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_feats_lists = self.extract_img_feat(img, img_metas)
        #print(img_feats_lists[0].shape, img_feats_lists[3].shape)

        #img_mod1 = img_feats_lists[0].transpose(2, 3)
        img_mod1 = img_feats_lists[0]
        B, C, H, W = img_mod1.shape[0], img_mod1.shape[1], img_mod1.shape[2], img_mod1.shape[3]
        img_mod2 = torch.reshape(img_mod1, (B, C, H*2, int(W/2)))
        #print(img_mod2.shape)
        pts_feats = self.extract_pts_feat(points, img_feats_lists, img_metas)

        # 双线性图片与点云特征尺寸相同
        if img_mod2.shape[2:] != pts_feats[0].shape[2:]:
            img_feats_tofuse = F.interpolate(img_mod2, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)
        else:
            img_feats_tofuse = img_mod2
        fusion_feature = self.fusion_block(pts_feats[0], img_feats_tofuse)
        return (img_feats_lists, [fusion_feature])

@DETECTORS.register_module()
class DynamicMVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNN, self).__init__(**kwargs)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
