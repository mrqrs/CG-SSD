import numba
import numpy as np
import torch
from .heatmap_utils import *
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.utils.box_utils import boxes_to_corners_3d
# import matplotlib.pyplot as plt
import math

class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

class CornerCoder(object):
    def __init__(self, point_cloud_range, voxel_size, out_size_factor, code_size=7,
                 use_log=False, with_velo=False):
        self.code_size = code_size
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_log = use_log
        self.with_velo = with_velo
    def max_points_num_corner_encode(self, points, box_corners, bboxes, heatmap, reg_target_map, min_radius,
                                   min_overlap, gt_label, cls_label):
        ### top y right x
        device = points.device
        # keep_corners = np.zeros((bboxes.shape[0], 2))
        # reg_target = np.zeros((bboxes.shape[0], reg_target_map.shape[-1]))

        keep_corners = torch.zeros((bboxes.shape[0], 2), device=device)
        reg_target = torch.zeros((bboxes.shape[0], reg_target_map.shape[-1]), device=device)
        # corner_index = torch.tensor([[3, 1],
        #                          [2, 0],
        #                          [1, 3],
        #                          [0, 2]], device=device)
        points_in_bboxes_index = points_in_boxes_gpu(points[:, :3].unsqueeze(0), bboxes.unsqueeze(0)).squeeze()

        # reg_target, keep_corners = max_points_encode_jit(points_in_bboxes_index.data.cpu().numpy(), points.cpu().numpy(), bboxes.cpu().numpy(), 
        #                                                     keep_corners, box_corners.cpu().numpy(), reg_target, self.use_log, self.with_velo)
        # reg_target = torch.tensor(reg_target, device=device)
        # keep_corners = torch.tensor(keep_corners, device=device)
        for i in range(bboxes.shape[0]):
            points_in_cur_bbox = points[points_in_bboxes_index==i]
            cur_bbox_rotation_angle = bboxes[i, 6]
            points_in_car_coor, center_in_car_coor = \
                velo_to_car_gpu(points_in_cur_bbox[:, :2], bboxes[i, :2],
                            -cur_bbox_rotation_angle)
            # np.savetxt('/opt/data/private/points_in_car_coor.txt', points_in_car_coor.data.cpu().numpy())
            relative_c_p = points_in_car_coor - center_in_car_coor
            q1 = (relative_c_p[:, 0] > 0) & (relative_c_p[:, 1] > 0)
            q2 = (relative_c_p[:, 0] > 0) & (relative_c_p[:, 1] < 0)
            q3 = (relative_c_p[:, 0] < 0) & (relative_c_p[:, 1] < 0)
            q4 = (relative_c_p[:, 0] < 0) & (relative_c_p[:, 1] > 0)
            number_per_3q = torch.tensor(
                [q1.sum() + q2.sum() + q4.sum(), q2.sum() + q1.sum() + q3.sum(), q3.sum() + q2.sum() + q4.sum(),
                 q4.sum() + q1.sum() + q3.sum()], device=points.device).reshape((-1, 1))
            number_per_q = torch.tensor([q1.sum(), q2.sum(), q3.sum(), q4.sum()], device=points.device).reshape((-1, 1))
            max_sum = torch.argmax(number_per_3q)
            max_q = torch.argmax(number_per_q)
            index_sum = number_per_q > 0
            if index_sum.sum() > 2:
                max_1 = max_sum
            else:
                max_1 = max_q
            keep_corners[i, :2] = box_corners[i, max_1, :2]
            reg_target[i, 2] = bboxes[i, 2]
            reg_target[i, [8, 9]] = (bboxes[i, [0, 1]] - keep_corners[i])*2. / torch.sqrt(bboxes[i, [3]]**2 + bboxes[i, [4]]**2)
            reg_target[i, 6] = torch.sin(cur_bbox_rotation_angle)
            reg_target[i, 7] = torch.cos(cur_bbox_rotation_angle)
            if self.use_log:  ###l, w, h
                reg_target[i, [3, 4, 5]] = torch.log(bboxes[i, [3, 4, 5]])
            else:
                reg_target[i, [3, 4, 5]] = bboxes[i, [3, 4, 5]]
            if self.with_velo:
                reg_target[i, [10, 11]] = bboxes[i, [7, 8]]
        # np.savetxt('/opt/data/private/points.txt', points.data.cpu().numpy())
        # np.savetxt('/opt/data/private/car_corners0.txt', box_corners[:, 0, :].reshape((-1, 3)).data.cpu().numpy())
        # np.savetxt('/opt/data/private/car_corners1.txt', box_corners[:, 1, :].reshape((-1, 3)).data.cpu().numpy())
        # np.savetxt('/opt/data/private/car_corners2.txt', box_corners[:, 2, :].reshape((-1, 3)).data.cpu().numpy())
        # np.savetxt('/opt/data/private/car_corners3.txt', box_corners[:, 3, :].reshape((-1, 3)).data.cpu().numpy())
        #
        # np.savetxt('/opt/data/private/car_vis_corners.txt', keep_corners.reshape((-1, 2)).data.cpu().numpy())
        corners_index, reg_target = corners_in_image_index(keep_corners, self.point_cloud_range, self.voxel_size,
                                                              self.out_size_factor,
                                                              bboxes, reg_target)
        heatmap, reg_target_map = create_heatmap(bboxes, corners_index, heatmap, reg_target, reg_target_map,
                                                    self.voxel_size, self.out_size_factor, min_radius, min_overlap, gt_label,
                                                    cls_label)
        # plt.imshow(heatmap[0].data.cpu().numpy())
        # plt.show()
        return heatmap, reg_target_map

    def corner0_encode(self, points, box_corners, bboxes,
                       heatmap, reg_target_map, min_radius, min_overlap, gt_label, cls_label):
        device = points.device
        keep_corners = torch.zeros((bboxes.shape[0], 2), device=device)
        reg_target = torch.zeros((bboxes.shape[0], reg_target_map.shape[-1]), device=device)
        corner_index = torch.tensor([[3, 1],
                                 [2, 0],
                                 [1, 3],
                                 [0, 2]], device=device)
        for i in range(bboxes.shape[0]):
            keep_corners[i, :2] = box_corners[i, 0, :2]
            reg_target[i, 2] = bboxes[i, 2]
            reg_target[i, [8, 9]] = (bboxes[i, [0, 1]] - keep_corners[i])*2. / torch.sqrt(bboxes[i, [3]]**2 + bboxes[i, [4]]**2)
            reg_target[i, 6] = torch.sin(bboxes[i, 6])
            reg_target[i, 7] = torch.cos(bboxes[i, 6])
            if self.use_log:
                reg_target[i, [3, 4, 5]] = torch.log(bboxes[i, [3, 4, 5]])
            else:
                reg_target[i, [3, 4, 5]] = bboxes[i, [3, 4, 5]]
            if self.with_velo:
                reg_target[i, [10, 11]] = bboxes[i, [7, 8]]
        corners_index, reg_target = corners_in_image_index(keep_corners, self.point_cloud_range, self.voxel_size,
                                                           self.out_size_factor,
                                                           bboxes, reg_target)
        heatmap, reg_target_map = create_heatmap(bboxes, corners_index, heatmap, reg_target, reg_target_map,
                                                 self.voxel_size, self.out_size_factor, min_radius, min_overlap,
                                                 gt_label,
                                                 cls_label)
        return heatmap, reg_target_map

    def decode(self, box_preds):
        batch, H, W, code_size = box_preds.size()
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(box_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(box_preds.device)
        # ys = ys.view(1, H, W).repeat(batch, 1, 1).to(box_preds.device) + 0.5
        # xs = xs.view(1, H, W).repeat(batch, 1, 1).to(box_preds.device) + 0.5
        xs = xs * self.out_size_factor * self.voxel_size[0] + \
             self.point_cloud_range[0] + box_preds[..., 0]
        ys = self.point_cloud_range[1] + ys * self.out_size_factor * self.voxel_size[1] \
             + box_preds[..., 1]
        if self.use_log:
            box_preds[..., [3, 4, 5]] = torch.exp(box_preds[..., [3, 4, 5]])
        else:
            box_preds[..., [3, 4, 5]] = box_preds[..., [3, 4, 5]]

        rotation = torch.atan2(box_preds[..., 6], box_preds[..., 7])
        temp_bbox = torch.cat([xs.unsqueeze(-1).reshape(-1, 1),
                                         ys.unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 2].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 3].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 4].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 5].unsqueeze(-1).reshape(-1, 1),
                                         rotation.unsqueeze(-1).reshape(-1, 1)], dim=-1)
        box_corners = boxes_to_corners_3d(temp_bbox.data.cpu().numpy())
        box_corners = torch.tensor(box_corners[:, [0, 1, 2, 3], :2], device=temp_bbox.device)
        

        temp_bbox_center = temp_bbox[:, :2].unsqueeze(1)    
        cc_vector = box_corners - temp_bbox_center
        pred_vector = box_preds[..., [8, 9]].reshape(-1, 1, 2)
        cos_cc = (cc_vector[:, :, 0] * pred_vector[..., 0] + cc_vector[..., 1]*pred_vector[..., 1]) / (torch.norm(cc_vector, dim=2)*torch.norm(pred_vector, dim=2))
        index = torch.argmax(cos_cc, dim=1)
        index = torch.cat([torch.arange(0, index.shape[0]).to(index.device).reshape(-1, 1), index.reshape(-1, 1)], dim=1)
        select_corners = box_corners[index[:, 0], index[:, 1], :]
        # center = (select_corners + temp_bbox[:, :2]) / 2.
        center = select_corners
        center = center.reshape(batch, H, W, 2)
        pred_corners = temp_bbox[..., :3]
        if not self.with_velo:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         rotation.unsqueeze(-1)], dim=-1)
        else:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         rotation.unsqueeze(-1),
                                         box_preds[..., 10].unsqueeze(-1),
                                         box_preds[..., 11].unsqueeze(-1)], dim=-1)
        return batch_box_preds, pred_corners

    def decodev1(self, box_preds):
        batch, H, W, code_size = box_preds.size()
        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(box_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(box_preds.device)
        xs = xs * self.out_size_factor * self.voxel_size[0] + \
             self.point_cloud_range[0] + box_preds[..., 0]
        ys = self.point_cloud_range[1] + ys * self.out_size_factor * self.voxel_size[1] \
             + box_preds[..., 1]
        if self.use_log:
            box_preds[..., [3, 4, 5]] = torch.exp(box_preds[..., [3, 4, 5]])
        else:
            box_preds[..., [3, 4, 5]] = box_preds[..., [3, 4, 5]]
        box_preds[..., [3, 4, 5]] = torch.where(torch.isinf(box_preds[..., [3, 4, 5]]), torch.full_like(box_preds[..., [3, 4, 5]], 0.), box_preds[..., [3, 4, 5]])
        diag_vlaue = torch.sqrt(box_preds[..., [3]]**2 + box_preds[..., [4]]**2) / 2.
        diag_vlaue = torch.where(torch.isinf(diag_vlaue), torch.full_like(diag_vlaue, 0.), diag_vlaue)
        center = box_preds[..., [8, 9]] * diag_vlaue
        center[..., [0]] = center[..., [0]] + xs.unsqueeze(-1)
        center[..., [1]] = center[..., [1]] + ys.unsqueeze(-1)
        center = torch.where(torch.isinf(center), torch.full_like(center, 0.), center)
        rotation = torch.atan2(box_preds[..., 6], box_preds[..., 7])
        temp_bbox = torch.cat([xs.unsqueeze(-1).reshape(-1, 1),
                                ys.unsqueeze(-1).reshape(-1, 1),
                                box_preds[..., 2].unsqueeze(-1).reshape(-1, 1)], dim=-1)
        
        pred_corners = temp_bbox[..., :3]
        if not self.with_velo:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         rotation.unsqueeze(-1)], dim=-1)
        else:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         rotation.unsqueeze(-1),
                                         box_preds[..., 10].unsqueeze(-1),
                                         box_preds[..., 11].unsqueeze(-1)], dim=-1)
        return batch_box_preds, pred_corners

    def decode_for_iou(self, box_pred):
        box_preds = box_pred.clone().detach()
        xs = box_preds[..., 0]
        ys = box_preds[..., 1]
        if self.use_log:
            box_preds[..., [3, 4, 5]] = torch.exp(box_preds[..., [3, 4, 5]])
        else:
            box_preds[..., [3, 4, 5]] = box_preds[..., [3, 4, 5]]

        rotation = torch.atan2(box_preds[..., 6], box_preds[..., 7])
        temp_bbox = torch.cat([xs.unsqueeze(-1).reshape(-1, 1),
                                         ys.unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 2].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 3].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 4].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 5].unsqueeze(-1).reshape(-1, 1),
                                         rotation.unsqueeze(-1).reshape(-1, 1)], dim=-1)
        box_corners = boxes_to_corners_3d(temp_bbox.data.cpu().numpy())
        box_corners = torch.tensor(box_corners[:, [0, 1, 2, 3], :2], device=temp_bbox.device)
        # temp_bbox_bev = center_to_corner_box2d(temp_bbox[:, :2].data.cpu().numpy(), temp_bbox[:, [3, 4]].data.cpu().numpy(),
        #                                              temp_bbox[:, 6].data.cpu().numpy())
        # temp_bbox_bev = torch.tensor(temp_bbox_bev[:, [3, 2, 1, 0], :], device=box_preds.device)

        temp_bbox_center = temp_bbox[:, :2].unsqueeze(1)
        cc_vector = box_corners - temp_bbox_center
        pred_vector = box_preds[..., [8, 9]].reshape(-1, 1, 2)
        cos_cc = (cc_vector[:, :, 0] * pred_vector[..., 0] + cc_vector[..., 1]*pred_vector[..., 1]) / (torch.norm(cc_vector, dim=2)*torch.norm(pred_vector, dim=2))
        index = torch.argmax(cos_cc, dim=1)
        index = torch.cat([torch.arange(0, index.shape[0]).to(index.device).reshape(-1, 1), index.reshape(-1, 1)], dim=1)
        select_corners = box_corners[index[:, 0], index[:, 1], :]
        # center = (select_corners + temp_bbox[:, :2]) / 2.
        center = select_corners
        if not self.with_velo:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         rotation.unsqueeze(-1)], dim=-1)
        else:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         rotation.unsqueeze(-1),
                                         box_preds[..., 10].unsqueeze(-1),
                                         box_preds[..., 11].unsqueeze(-1)], dim=-1)
        return batch_box_preds
    def decode_for_center(self, box_preds):
        xs = box_preds[..., 0]
        ys = box_preds[..., 1]
        if self.use_log:
            box_preds[..., [3, 4, 5]] = torch.exp(box_preds[..., [3, 4, 5]])
        else:
            box_preds[..., [3, 4, 5]] = box_preds[..., [3, 4, 5]]
        
        rotation = torch.atan2(box_preds[..., 6], box_preds[..., 7])
        
        temp_bbox = torch.cat([xs.unsqueeze(-1).reshape(-1, 1),
                                         ys.unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 2].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 3].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 4].unsqueeze(-1).reshape(-1, 1),
                                         box_preds[..., 5].unsqueeze(-1).reshape(-1, 1),
                                         rotation.unsqueeze(-1).reshape(-1, 1)], dim=-1)
        box_corners = boxes_to_corners_3d(temp_bbox.data.cpu().numpy())
        box_corners = torch.tensor(box_corners[:, [0, 1, 2, 3], :2], device=temp_bbox.device)
        temp_bbox_center = temp_bbox[:, :2].unsqueeze(1)
        cc_vector = box_corners - temp_bbox_center
        pred_vector = box_preds[..., [8, 9]].reshape(-1, 1, 2)
        cos_cc = (cc_vector[:, :, 0] * pred_vector[..., 0] + cc_vector[..., 1]*pred_vector[..., 1]) / (torch.norm(cc_vector, dim=2)*torch.norm(pred_vector, dim=2))
        
        index = torch.argmax(cos_cc, dim=1)
        index = torch.cat([torch.arange(0, index.shape[0]).to(index.device).reshape(-1, 1), index.reshape(-1, 1)], dim=1)
        select_corners = box_corners[index[:, 0], index[:, 1], :]
        center = select_corners
        
        if not self.with_velo:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         rotation.unsqueeze(-1)], dim=-1)
        else:
            batch_box_preds = torch.cat([center[..., 0].unsqueeze(-1),
                                         center[..., 1].unsqueeze(-1),
                                         box_preds[..., 2].unsqueeze(-1),
                                         box_preds[..., 4].unsqueeze(-1),
                                         box_preds[..., 5].unsqueeze(-1),
                                         box_preds[..., 3].unsqueeze(-1),
                                         rotation.unsqueeze(-1),
                                         box_preds[..., 10].unsqueeze(-1),
                                         box_preds[..., 11].unsqueeze(-1)], dim=-1)
        # batch_box_preds = torch.where(torch.isnan(batch_box_preds), torch.full_like(batch_box_preds, 0.), batch_box_preds)
        # batch_box_preds = torch.where(torch.isinf(batch_box_preds), torch.full_like(batch_box_preds, 0.), batch_box_preds)
        return batch_box_preds

@numba.jit(nopython=True)
def max_points_encode_jit(points_in_bboxes_index, points, bboxes, keep_corners, box_corners, reg_target, use_log, with_velo):
    for i in range(bboxes.shape[0]):
        points_in_cur_bbox = points[points_in_bboxes_index==i]
        cur_bbox_rotation_angle = bboxes[i, 6]
        # points_in_car_coor, center_in_car_coor = \
        #     velo_to_car_gpu(points_in_cur_bbox[:, :2], bboxes[i, :2],
        #                 -cur_bbox_rotation_angle)
        rotation_matrix = np.array([[np.cos(cur_bbox_rotation_angle), np.sin(cur_bbox_rotation_angle)],
                                            [-np.sin(cur_bbox_rotation_angle), np.cos(cur_bbox_rotation_angle)]])
        points_in_cur_bbox = points_in_cur_bbox[:, 1:3]
        center_in_velo = bboxes[i, :2]
        points_in_car_coor = points_in_cur_bbox @ rotation_matrix
        center_in_car_coor = center_in_velo @ rotation_matrix
        # np.savetxt('/opt/data/private/points_in_car_coor.txt', points_in_car_coor.data.cpu().numpy())
        relative_c_p = points_in_car_coor - center_in_car_coor
        q1 = (relative_c_p[:, 0] > 0) & (relative_c_p[:, 1] > 0)
        q2 = (relative_c_p[:, 0] > 0) & (relative_c_p[:, 1] < 0)
        q3 = (relative_c_p[:, 0] < 0) & (relative_c_p[:, 1] < 0)
        q4 = (relative_c_p[:, 0] < 0) & (relative_c_p[:, 1] > 0)
        number_per_3q = np.array(
            [q1.sum() + q2.sum() + q4.sum(), q2.sum() + q1.sum() + q3.sum(), q3.sum() + q2.sum() + q4.sum(),
            q4.sum() + q1.sum() + q3.sum()]).reshape((-1, 1))
        number_per_q = np.array([q1.sum(), q2.sum(), q3.sum(), q4.sum()]).reshape((-1, 1))
        max_sum = np.argmax(number_per_3q)
        max_q = np.argmax(number_per_q)
        index_sum = number_per_q > 0
        if index_sum.sum() > 2:
            max_1 = max_sum
        else:
            max_1 = max_q
        keep_corners[i, :2] = box_corners[i, max_1, :2]
        reg_target[i, 2] = bboxes[i, 2]
        reg_target[i, 8] = (bboxes[i, 0] - keep_corners[i, 0]) / np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2)
        reg_target[i, 9] = (bboxes[i, 1] - keep_corners[i, 1]) / np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2)
        reg_target[i, 6] = np.sin(cur_bbox_rotation_angle)
        reg_target[i, 7] = np.cos(cur_bbox_rotation_angle)
        if use_log:  ###l, w, h
            reg_target[i, 3] = np.log(bboxes[i, 3])
            reg_target[i, 4] = np.log(bboxes[i, 4])
            reg_target[i, 5] = np.log(bboxes[i, 5])
        else:
            reg_target[i, 3] = bboxes[i, 3]
            reg_target[i, 4] = bboxes[i, 4]
            reg_target[i, 5] = bboxes[i, 5]
        if with_velo:
            reg_target[i, 10] = bboxes[i, 7]
            reg_target[i, 11] = bboxes[i, 8]
    return reg_target, keep_corners
            
class LidarRcnnCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1
    
    def encode_torch(self, anchors, boxes):
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)
        # diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / dxa
        yt = (yg - ya) / dya
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        # diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * dxa + xa
        yg = yt * dya + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

class LidarRcnnCoderV1(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1
    
    def encode_torch(self, anchors, boxes):
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)
        # diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = xg - xa
        yt = yg - ya
        zt = zg - za
        dxt = dxg - dxa
        dyt = dyg - dya
        dzt = dzg - dza
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        # diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt + xa
        yg = yt + ya
        zg = zt + za

        dxg = dxt + dxa
        dyg = dyt + dya
        dzg = dzt + dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

