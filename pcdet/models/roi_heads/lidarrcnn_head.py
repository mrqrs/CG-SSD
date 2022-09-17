import torch.nn as nn
import torch
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
import numpy as np
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.ops.roipoint_pool3d.roipoint_pool3d_utils import RoIPointPool3d
from pcdet.utils.box_utils import enlarge_box3d, boxes_to_corners_3d
from ...ops.iou3d_nms import iou3d_nms_cuda


class LIDARRCNNHead(RoIHeadTemplate):
    def __init__(self, model_cfg, num_class=1, input_channels=9):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.model_cfg = model_cfg
        self.feat = PointNetfeat(model_cfg.INPUT_CHANNELS, model_cfg.X)
        self.fc1 = nn.Linear(512 * model_cfg.X, 256 * model_cfg.X)
        self.fc2 = nn.Linear(256 * model_cfg.X, 256)

        self.pre_bn = nn.BatchNorm1d(model_cfg.INPUT_CHANNELS)
        self.bn1 = nn.BatchNorm1d(256 * model_cfg.X)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc_s1 = nn.Linear(256, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        # self.fc_c1 = nn.Linear(256, 256)
        # self.fc_c2 = nn.Linear(256, num_class, bias=False)
        self.fc_ce1 = nn.Linear(256, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(256, 256)
        self.fc_hr2 = nn.Linear(256, 1, bias=False)
        self.point_number = model_cfg.POINT_NUMBER
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)


    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # batch_dict['reg_valid_mask']  = targets_dict['reg_valid_mask']
            batch_dict, targets_dict = self.get_inputs(batch_dict, targets_dict)
        else:
            batch_dict = self.get_inputs_test(batch_dict)

        
        x = batch_dict['inputs']
        x = x.view(-1, self.point_number, self.model_cfg.INPUT_CHANNELS)
        #print("After proposal layer")
        #print(targets_dict['rois'].shape)
        x = x.transpose(2, 1)
        x = self.feat(self.pre_bn(x))
        x = F.relu(self.bn1(self.fc1(x)))
        feat = F.relu(self.bn2(self.fc2(x)))

        # x = F.relu(self.fc_c1(feat))
        # logits = self.fc_c2(x)

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)
        rcnn_reg = torch.cat((centers, sizes, headings), dim=-1)
        if self.training:
            # targets_dict['rcnn_cls'] = logits
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        else:
            # batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            #     batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=logits, box_preds=rcnn_reg
            # )
            batch_dict = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=batch_dict['roi_scores'], box_preds=rcnn_reg, batch_dict=batch_dict
            )

        return batch_dict

    def get_inputs(self, batch_dict, target_dict):
        batch_size = batch_dict['batch_size']
        batch_rois = target_dict['rois']
        reg_valid_mask = target_dict['reg_valid_mask']
        rcnn_cls_labels = target_dict['rcnn_cls_labels']
        batch_points = batch_dict['points']
        pool = RoIPointPool3d(self.point_number, (0.3, 0.3, 0.3))
        batch_inputs = batch_rois.new_zeros(batch_size, batch_rois.shape[1], self.point_number, self.model_cfg.INPUT_CHANNELS)
        for i in range(batch_size):
            rois = batch_rois[i]
            points = batch_points[batch_points[:, 0]==i]
            pooled_features, pooled_empty_flag = pool(points[:, 1:4].unsqueeze(0), points[:, 1:].unsqueeze(0), rois.unsqueeze(0))

            batch_rois[i] = rois
            if self.model_cfg.INPUT_CHANNELS==9:
                pooled_features = pooled_features[..., 3:6].squeeze(0)
            else:
                pooled_features = pooled_features[..., 3:].squeeze(0)

            pooled_empty_flag = pooled_empty_flag.squeeze(0)
            rois = rois.unsqueeze(1)
            point_set = pooled_features
            point_set[..., 2] = pooled_features[..., 2] - rois[..., 2]
            point_set[..., :2] = point_set[..., :2] - rois[..., :2]
            point_set = common_utils.rotate_points_along_z(
                    point_set, -rois[..., 6].view(-1)
                )
            point_set = torch.cat([point_set, -point_set[..., :3] + rois[..., [3, 4, 5]] / 2., point_set[..., :3] + rois[..., [3, 4, 5]] / 2.], dim=-1)

            pooled_empty_flag = pooled_empty_flag==1
            rcnn_cls_labels[i, pooled_empty_flag] = 0
            reg_valid_mask[i, pooled_empty_flag] = 0
            point_set[pooled_empty_flag] = 0.
            batch_inputs[i] = point_set
            # points = batch_points[batch_points[:, 0]==i]
            # points_in_rois = points_in_boxes_gpu(points[:, 1:4].unsqueeze(0), rois.unsqueeze(0)).squeeze()
            # for j in range(rois.shape[0]):
            #     point_in_roi_index = points_in_rois == j
            #     if point_in_roi_index.sum()==0:
            #         rcnn_cls_labels[i, j] = 0
            #         reg_valid_mask[i, j] = 0
            #         continue
            #     points_in_roi = points[point_in_roi_index][:, 1:]
            #     points_in_roi[:, 2] -= rois[j, 2]
            #     choice = np.random.choice(points_in_roi.shape[0], self.point_number, replace=True)
            #     point_set = points_in_roi[choice, :3]
            #     # transform the points to pred box coordinate
            #     norm_xy = point_set[:, :2] - rois[j, :2]
            #     point_set[:, :2] = torch.mm(norm_xy, rotz(rois[j, -1]).to(rois.device))
            #     point_set = torch.cat([
            #         point_set, -point_set[:, :3] + rois[j, [3, 4, 5]] / 2,
            #         rois[j, [3, 4, 5]] / 2 + point_set[:, :3]
            #     ], dim=-1)
            #     batch_inputs[i, j, ...] = point_set
        batch_dict['inputs'] = batch_inputs
        target_dict['reg_valid_mask'] = reg_valid_mask
        target_dict['rcnn_cls_labels'] = rcnn_cls_labels
        return batch_dict, target_dict
    
    def get_inputs_test(self, batch_dict):
        batch_size = batch_dict['batch_size']
        batch_rois = batch_dict['rois']
        batch_points = batch_dict['points']
        pool = RoIPointPool3d(self.point_number, (0.3, 0.3, 0.3))
        batch_inputs = batch_rois.new_zeros(batch_size, batch_rois.shape[1], self.point_number, self.model_cfg.INPUT_CHANNELS)
        for i in range(batch_size):
            rois = batch_rois[i]
            points = batch_points[batch_points[:, 0]==i]
            pooled_features, pooled_empty_flag = pool(points[:, 1:4].unsqueeze(0), points[:, 1:].unsqueeze(0), rois.unsqueeze(0))

            batch_rois[i] = rois
            if self.model_cfg.INPUT_CHANNELS==9:
                pooled_features = pooled_features[..., 3:6].squeeze(0)
            else:
                pooled_features = pooled_features[..., 3:].squeeze(0)

            pooled_empty_flag = pooled_empty_flag.squeeze(0)
            rois = rois.unsqueeze(1)
            point_set = pooled_features
            point_set[..., 2] = pooled_features[..., 2] - rois[..., 2]
            point_set[..., :2] = point_set[..., :2] - rois[..., :2]
            point_set = common_utils.rotate_points_along_z(
                    point_set, -rois[..., 6].view(-1)
                )
            
            point_set = torch.cat([point_set, -point_set[..., :3] + rois[..., [3, 4, 5]] / 2., point_set[..., :3] + rois[..., [3, 4, 5]] / 2.], dim=-1)

            pooled_empty_flag = pooled_empty_flag==1
            point_set[pooled_empty_flag] = 0.
            batch_inputs[i] = point_set
            # points = batch_points[batch_points[:, 0]==i]
            # points_in_rois = points_in_boxes_gpu(points[:, 1:4].unsqueeze(0), rois.unsqueeze(0)).squeeze()
            # for j in range(rois.shape[0]):
            #     point_in_roi_index = points_in_rois == j
            #     if point_in_roi_index.sum()==0:
            #         rcnn_cls_labels[i, j] = 0
            #         reg_valid_mask[i, j] = 0
            #         continue
            #     points_in_roi = points[point_in_roi_index][:, 1:]
            #     points_in_roi[:, 2] -= rois[j, 2]
            #     choice = np.random.choice(points_in_roi.shape[0], self.point_number, replace=True)
            #     point_set = points_in_roi[choice, :3]
            #     # transform the points to pred box coordinate
            #     norm_xy = point_set[:, :2] - rois[j, :2]
            #     point_set[:, :2] = torch.mm(norm_xy, rotz(rois[j, -1]).to(rois.device))
            #     point_set = torch.cat([
            #         point_set, -point_set[:, :3] + rois[j, [3, 4, 5]] / 2,
            #         rois[j, [3, 4, 5]] / 2 + point_set[:, :3]
            #     ], dim=-1)
            #     batch_inputs[i, j, ...] = point_set
        batch_dict['inputs'] = batch_inputs
        # batch_dict['rois'] = batch_rois
        return batch_dict
        
    def jitter(self, bbox, thres=0.5):
        for i in range(bbox.shape[0]):
            if 0<bbox[i, -1]<3:
                if np.random.rand() < thres:
                    return bbox
                range_config = [[0.2, 0.1, np.pi / 12, 0.7], [0.3, 0.15, np.pi / 12, 0.6],
                                [0.5, 0.15, np.pi / 9, 0.5], [0.8, 0.15, np.pi / 6, 0.3],
                                [1.0, 0.15, np.pi / 3, 0.2]]
                idx = np.random.randint(low=0, high=len(range_config), size=(1, ))[0]

                pos_shift = ((torch.rand(3) - 0.5) / 0.5) * range_config[idx][0]
                hwl_scale = ((torch.rand(3) - 0.5) / 0.5) * range_config[idx][1] + 1.0
                angle_rot = ((torch.rand(1) - 0.5) / 0.5) * range_config[idx][2]

                bbox[i] = torch.cat(
                    [bbox[i, 0:3] + pos_shift.to(bbox.device), bbox[i, 3:6] * hwl_scale.to(bbox.device), bbox[i, 6:7] + angle_rot.to(bbox.device)])
        return bbox
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict
    
    def _nms_gpu_3d(self, boxes, scores, thresh, pre_maxsize=None, post_max_size = None):
        """
        :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        :param scores: (N)
        :param thresh:
        :return:
        """
        assert boxes.shape[1] == 7
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]

        boxes = boxes[order].contiguous()
        keep = torch.LongTensor(boxes.size(0))
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
        selected =  order[keep[:num_out].cuda()].contiguous()

        if post_max_size is not None:
            selected = selected[:post_max_size]

        return selected
    
    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds, batch_dict):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, 1)
        batch_box_preds = box_preds.view(batch_size, -1, code_size)
        roi_ry = rois[:, :, 6].view(-1)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        if self.model_cfg.LOSS_CONFIG.REG_LOSS == 'l1':
            batch_box_preds[..., [0, 1, 2, 6]] += local_rois[..., [0, 1, 2, 6]]
            batch_box_preds = batch_box_preds.view(-1, code_size)
        else:
            roi_ry = rois[:, :, 6].view(-1)
            batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        pred_dicts = []
        for i in range(batch_size):
            cur_batch_pred_scores = batch_cls_preds[i]
            cur_batch_pred_boxes = batch_box_preds[i]
            cur_batch_pred_labels = batch_dict['roi_labels'][i]
            cur_batch_boxes = []
            cur_batch_scores = []
            cur_batch_labels = []
            for j in range(3):
                selected = cur_batch_pred_labels==(j+1)
                if selected.sum() == 0:
                    continue
                boxes = cur_batch_pred_boxes[selected]
                scores = cur_batch_pred_scores[selected]
                labels = cur_batch_pred_labels[selected]
                selected = (scores >= 0.3).squeeze()
                if selected.sum() == 0:
                    continue
                boxes = boxes[selected]
                scores = scores[selected]
                labels = labels[selected]
                selected = self._nms_gpu_3d(boxes, scores, 0.01, 500, 128).squeeze()
                cur_batch_boxes.append(boxes[selected].reshape(-1, 7))
                cur_batch_scores.append(scores[selected].reshape(-1, 1))
                cur_batch_labels.append(labels[selected].reshape(-1, 1))
            if len(cur_batch_boxes)>0:
                cur_batch_boxes = torch.cat(cur_batch_boxes, dim=0)
                cur_batch_scores = torch.cat(cur_batch_scores)
                cur_batch_labels = torch.cat(cur_batch_labels)
                record_dict = {
                    "pred_boxes": cur_batch_boxes,
                    "pred_scores": cur_batch_scores.reshape(-1),
                    "pred_labels": cur_batch_labels.reshape(-1).long()
                    }
            else:
                cls_preds = torch.empty(0, 1)
                reg_preds = torch.empty(0, 7)
                label = torch.empty(0)
                record_dict = {
                "pred_boxes": reg_preds,
                "pred_scores": cls_preds,
                "pred_labels": label
                }
            pred_dicts.append(record_dict)
        batch_dict['pred_dicts'] = pred_dicts
        batch_dict['has_class_labels'] = True  # Force to be true
        return batch_dict


class PointNetfeat(nn.Module):
    def __init__(self, pts_dim, x=1):
        super(PointNetfeat, self).__init__()
        self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(pts_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(64 * x)
        self.bn2 = nn.BatchNorm1d(128 * x)
        self.bn3 = nn.BatchNorm1d(self.output_channel)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # NOTE: should not put a relu
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.output_channel)
        return x
        
def rotz(t):
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s], [s, c]])
