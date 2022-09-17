import numpy as np
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from .target_assigner.center_assigner import CenterAssigner_Aux
from ...utils import box_coder_utils, common_utils, loss_utils

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, voxel_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

class AnchorHeadSingle_Aux(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, voxel_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.share_aux = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.heatmap0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_class, 3, 1, 1)
        )
        self.heatmap1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_class, 3, 1, 1)
        )
        self.heatmap2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_class, 3, 1, 1)
        )

        self.offset0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )
        self.offset1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )
        self.offset2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )

        self.conv_cls = nn.Conv2d(
            input_channels+21, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels+21, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels+21,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.aux_target = CenterAssigner_Aux(model_cfg.TARGET_ASSIGNER_CONFIG_AUX, self.num_class, True, grid_size, point_cloud_range, voxel_size)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        for m in self.share_aux:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -np.log((1 - pi) / pi))
                nn.init.normal_(m.weight, mean=0, std=0.001)

        for m in self.heatmap0:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -np.log((1 - pi) / pi))
                nn.init.normal_(m.weight, mean=0, std=0.001)
        for m in self.heatmap1:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -np.log((1 - pi) / pi))
                nn.init.normal_(m.weight, mean=0, std=0.001)
        for m in self.heatmap2:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -np.log((1 - pi) / pi))
                nn.init.normal_(m.weight, mean=0, std=0.001)

        for m in self.heatmap0:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -np.log((1 - pi) / pi))
                nn.init.normal_(m.weight, mean=0, std=0.001)
        for m in self.heatmap1:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -np.log((1 - pi) / pi))
                nn.init.normal_(m.weight, mean=0, std=0.001)
        for m in self.heatmap2:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, -np.log((1 - pi) / pi))
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        aux_share_features = self.share_aux(spatial_features_2d)
        heatmaps = []
        offsets = []
        heatmaps.append(self.heatmap0(aux_share_features))
        heatmaps.append(self.heatmap1(aux_share_features))
        heatmaps.append(self.heatmap2(aux_share_features))

        offsets.append(self.offset0(aux_share_features))
        offsets.append(self.offset1(aux_share_features))
        offsets.append(self.offset2(aux_share_features))

        heatmap_features = torch.cat(heatmaps, dim=1)
        offset_features = torch.cat(offsets, dim=1)
        spatial_features_2d = torch.cat([spatial_features_2d, heatmap_features, offset_features], dim=1)
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['heatmaps'] = heatmaps
        self.forward_ret_dict['offsets'] = offsets

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            aux_target_dict = self.aux_target.assign_targets_3p(data_dict['gt_boxes'], data_dict['points'], [False, False, False, True, True])
            targets_dict.update(aux_target_dict)
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def build_losses(self, losses_cfg):
        if self.model_cfg.get('USE_BEV_SEG', False):
            self.add_module(
                'seg_loss_func',
                loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
            )

        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
        self.add_module(
            'crit',
            loss_utils.CenterNetFocalLoss()
        )
        self.add_module(
            'crit_reg',
            loss_utils.CenterNetRegLoss()
        )

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)

        if self.model_cfg.get('USE_BEV_SEG', False):
            seg_loss, tb_dict_seg = self.get_seg_layer_loss()
            tb_dict.update(tb_dict_seg)
            rpn_loss = cls_loss + box_loss + seg_loss
        else:
            rpn_loss = cls_loss + box_loss
        aux_pred_heatmap_dict = self.forward_ret_dict['heatmaps']
        aux_pred_heatmap_dict[0] = self._sigmoid(aux_pred_heatmap_dict[0])
        aux_pred_heatmap_dict[1] = self._sigmoid(aux_pred_heatmap_dict[1])
        aux_pred_heatmap_dict[2] = self._sigmoid(aux_pred_heatmap_dict[2])
        hm0_aux_loss = self.crit(aux_pred_heatmap_dict[0], self.forward_ret_dict['heatmap0'][0])
        hm1_aux_loss = self.crit(aux_pred_heatmap_dict[1], self.forward_ret_dict['heatmap1'][0])
        hm2_aux_loss = self.crit(aux_pred_heatmap_dict[2], self.forward_ret_dict['heatmap2'][0])

        aux_pred_dict = self.forward_ret_dict['offsets']
        target_box_offset0_encoding = self.forward_ret_dict['offset0'][0]
        pred_corner0_offset_encoding = aux_pred_dict[0]

        offset0_aux_loss = self.crit_reg(
            pred_corner0_offset_encoding,
            self.forward_ret_dict['mask0'][0],
            self.forward_ret_dict['ind0'][0],
            target_box_offset0_encoding
        )

        target_box_offset1_encoding = self.forward_ret_dict['offset1'][0]
        pred_corner1_offset_encoding = aux_pred_dict[1]

        offset1_aux_loss = self.crit_reg(
            pred_corner1_offset_encoding,
            self.forward_ret_dict['mask1'][0],
            self.forward_ret_dict['ind1'][0],
            target_box_offset1_encoding
        )

        target_box_offset2_encoding = self.forward_ret_dict['offset2'][0]
        pred_corner2_offset_encoding = aux_pred_dict[2]

        offset2_aux_loss = self.crit_reg(
            pred_corner2_offset_encoding,
            self.forward_ret_dict['mask2'][0],
            self.forward_ret_dict['ind2'][0],
            target_box_offset2_encoding
        )
        tb_dict['hm0_loss'] = hm0_aux_loss.item()
        tb_dict['hm1_loss'] = hm1_aux_loss.item()
        tb_dict['hm2_loss'] = hm2_aux_loss.item()
        tb_dict['offset0_loss'] = offset0_aux_loss.sum().item()
        tb_dict['offset1_loss'] = offset1_aux_loss.sum().item()
        tb_dict['offset2_loss'] = offset2_aux_loss.sum().item()
        rpn_loss += 0.25*(hm0_aux_loss + hm1_aux_loss + hm2_aux_loss + offset0_aux_loss.sum() + offset1_aux_loss.sum() + offset2_aux_loss.sum())
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict
    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y