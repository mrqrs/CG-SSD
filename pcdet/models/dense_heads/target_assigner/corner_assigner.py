import torch
from ....ops.center_ops import center_ops_cuda
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.utils.box_utils import boxes_to_corners_3d
from pcdet.utils.heatmap_utils import velo_to_car_gpu, velo_to_car_cpu
import numba
import numpy as np
import math


class CornerAssigner(object):
    def __init__(self, assigner_cfg, use_corner0, num_classes, no_log, grid_size, pc_range, voxel_size):
        """Return cornernet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.num_classes = num_classes
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        # self.class_to_idx = assigner_cfg.mapping
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log
        self.use_corner0 = use_corner0
        self.tasks = assigner_cfg.tasks

    def assign_targets(self, gt_boxes, points, encode_size):
        """
        Args:
            gt_boxes: (B, M, C + cls)
        Returns:
        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor # grid_size WxHxD feature_map_size WxH
        batch_size = gt_boxes.shape[0]
        code_size = gt_boxes.shape[2] #cls -> sin/cos
        num_classes = self.num_classes
        assert gt_boxes[:, :, -1].max().item() <= num_classes, "labels must match, found {}".format(gt_boxes[:, :, -1].max().item())
        
        gt_corners = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 2), device=gt_boxes.device)
        # gt_corners = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1]*3, 2), device=gt_boxes.device)
        for b in range(batch_size):
            bboxes = gt_boxes[b]
            box_corners = boxes_to_corners_3d(bboxes[:, :7].data.cpu().numpy())
            # box_corners = torch.tensor(box_corners[:, [0, 1, 2, 3], :], device=gt_boxes.device)
            box_corners = box_corners[:, [0, 1, 2, 3], :]
            cur_points = points[points[:, 0]==b]
            points_in_bboxes_index = points_in_boxes_gpu(cur_points[:, 1:4].unsqueeze(0), bboxes[:, :7].unsqueeze(0)).squeeze()
            index = bboxes[:, -1] > 0
            cur_gt_corners = get_corner(index.sum().item(), bboxes.cpu().numpy(), cur_points.cpu().numpy(), points_in_bboxes_index.cpu().numpy(),
                        gt_corners[b].cpu().numpy(), box_corners, b, self.use_corner0)
            # cur_gt_corners = get_cornerV1(index.sum().item(), bboxes.cpu().numpy(), cur_points.cpu().numpy(), points_in_bboxes_index.cpu().numpy(),
            #             gt_corners[b].cpu().numpy(), box_corners, b, self.use_corner0)
            gt_corners[b] = torch.tensor(cur_gt_corners, device=gt_corners.device)
            # for i in range(index.sum()):
            #     if self.use_corner0[int(bboxes[i, -1]-1)]:
            #         gt_corners[b, i, :] = box_corners[i, 0, :2]
            #     else:
            #         points_in_cur_bbox = cur_points[points_in_bboxes_index==i]
            #         cur_bbox_rotation_angle = bboxes[i, 6]
            #         points_in_car_coor, center_in_car_coor = \
            #             velo_to_car_gpu(points_in_cur_bbox[:, 1:3], bboxes[i, :2], -cur_bbox_rotation_angle)
            #         relative_c_p = points_in_car_coor - center_in_car_coor
            #         q1 = (relative_c_p[:, 0] > 0) & (relative_c_p[:, 1] > 0)
            #         q2 = (relative_c_p[:, 0] > 0) & (relative_c_p[:, 1] < 0)
            #         q3 = (relative_c_p[:, 0] < 0) & (relative_c_p[:, 1] < 0)
            #         q4 = (relative_c_p[:, 0] < 0) & (relative_c_p[:, 1] > 0)
            #         number_per_3q = torch.tensor(
            #             [q1.sum() + q2.sum() + q4.sum(), q2.sum() + q1.sum() + q3.sum(), q3.sum() + q2.sum() + q4.sum(),
            #             q4.sum() + q1.sum() + q3.sum()], device=points.device).reshape((-1, 1))
            #         number_per_q = torch.tensor([q1.sum(), q2.sum(), q3.sum(), q4.sum()], device=points.device).reshape((-1, 1))
            #         max_sum = torch.argmax(number_per_3q)
            #         max_q = torch.argmax(number_per_q)
            #         index_sum = number_per_q > 0
            #         if index_sum.sum() > 2:
            #             max_1 = max_sum
            #         else:
            #             max_1 = max_q
            #         gt_corners[b, i, :2] = box_corners[i, max_1, :2]
        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}

        heatmap = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        gt_ind = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, encode_size), dtype = torch.float32).to(gt_boxes.device)

        center_ops_cuda.draw_corner_gpu(gt_boxes, gt_corners, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        offset = 0
        for task_id in range(len(self.task_class)):
            end = offset + self.task_class[task_id]
            heatmap_of_task = heatmap[:, offset:end, :, :]
            gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, encode_size)
            gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_encoding_merged = torch.zeros((batch_size, max_objs, encode_size), dtype=torch.float32).to(gt_boxes.device)
            offset = end
            for i in range(batch_size):
                mask = gt_mask_of_task[i] == 1
                mask_range = mask.sum().item()
                assert mask_range <= max_objs
                gt_mask_merged[i, :mask_range] = gt_mask_of_task[i, mask]
                gt_ind_merged[i, :mask_range] = gt_ind_of_task[i, mask]
                gt_cat_merged[i, :mask_range] = gt_cat_of_task[i, mask]
                gt_box_encoding_merged[i, :mask_range, :] = gt_box_encoding_of_task[i, mask, :]
                # only perform log on valid gt_box_encoding
                if not self.no_log:
                    gt_box_encoding_merged[i, :mask_range, 3:6] = torch.log(gt_box_encoding_merged[i, :mask_range, 3:6]) # log(wlh)

            heatmaps[task_id] = heatmap_of_task
            gt_inds[task_id] = gt_ind_merged.long()
            gt_masks[task_id] = gt_mask_merged.bool()
            gt_cats[task_id] = gt_cat_merged.long()
            gt_box_encodings[task_id] = gt_box_encoding_merged

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict

    def assign_targets_v1(self, gt_boxes, points, encode_size):
        """
        Args:
            gt_boxes: (B, M, C + cls)
        Returns:
        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor # grid_size WxHxD feature_map_size WxH
        batch_size = gt_boxes.shape[0]
        code_size = gt_boxes.shape[2] #cls -> sin/cos
        num_classes = self.num_classes
        assert gt_boxes[:, :, -1].max().item() <= num_classes, "labels must match, found {}".format(gt_boxes[:, :, -1].max().item())
        
        gt_corners = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 10), device=gt_boxes.device)
        # gt_corners = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1]*3, 2), device=gt_boxes.device)
        for b in range(batch_size):
            bboxes = gt_boxes[b]
            box_corners = boxes_to_corners_3d(bboxes[:, :7].data.cpu().numpy())
            # box_corners = torch.tensor(box_corners[:, [0, 1, 2, 3], :], device=gt_boxes.device)
            box_corners = box_corners[:, [0, 1, 2, 3], :]
            cur_points = points[points[:, 0]==b]
            points_in_bboxes_index = points_in_boxes_gpu(cur_points[:, 1:4].unsqueeze(0), bboxes[:, :7].unsqueeze(0)).squeeze()
            index = bboxes[:, -1] > 0
            cur_gt_corners = get_corner_with_target(index.sum().item(), bboxes.cpu().numpy(), cur_points.cpu().numpy(), points_in_bboxes_index.cpu().numpy(),
                        gt_corners[b].cpu().numpy(), box_corners, b, self.use_corner0)
            gt_corners[b] = torch.tensor(cur_gt_corners, device=gt_corners.device)
        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}

        heatmap = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        gt_ind = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, encode_size), dtype = torch.float32).to(gt_boxes.device)

        center_ops_cuda.draw_corner_gpu_v1(gt_corners, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        offset = 0
        for task_id, task in enumerate(self.tasks):
            end = offset + len(task.class_names)
            heatmap_of_task = heatmap[:, offset:end, :, :]
            gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, encode_size)
            gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_encoding_merged = torch.zeros((batch_size, max_objs, encode_size), dtype=torch.float32).to(gt_boxes.device)
            offset = end
            for i in range(batch_size):
                mask = gt_mask_of_task[i] == 1
                mask_range = mask.sum().item()
                assert mask_range <= max_objs
                gt_mask_merged[i, :mask_range] = gt_mask_of_task[i, mask]
                gt_ind_merged[i, :mask_range] = gt_ind_of_task[i, mask]
                gt_cat_merged[i, :mask_range] = gt_cat_of_task[i, mask]
                gt_box_encoding_merged[i, :mask_range, :] = gt_box_encoding_of_task[i, mask, :]
                # only perform log on valid gt_box_encoding
                if not self.no_log:
                    gt_box_encoding_merged[i, :mask_range, 3:6] = torch.log(gt_box_encoding_merged[i, :mask_range, 3:6]) # log(wlh)

            heatmaps[task_id] = heatmap_of_task
            gt_inds[task_id] = gt_ind_merged.long()
            gt_masks[task_id] = gt_mask_merged.bool()
            gt_cats[task_id] = gt_cat_merged.long()
            gt_box_encodings[task_id] = gt_box_encoding_merged

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict

    def assign_targets_v2(self, gt_boxes, points, encode_size):
        """
        Args:
            gt_boxes: (B, M, C + cls)
        Returns:
        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor # grid_size WxHxD feature_map_size WxH
        batch_size = gt_boxes.shape[0]
        code_size = gt_boxes.shape[2] #cls -> sin/cos
        num_classes = self.num_classes
        assert gt_boxes[:, :, -1].max().item() <= num_classes, "labels must match, found {}".format(gt_boxes[:, :, -1].max().item())
        
        gt_corners = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 10), device=gt_boxes.device)
        for b in range(batch_size):
            bboxes = gt_boxes[b]
            box_corners = boxes_to_corners_3d(bboxes[:, :7].data.cpu().numpy())
            box_corners = box_corners[:, [0, 1, 2, 3], :]
            cur_points = points[points[:, 0]==b]
            points_in_bboxes_index = points_in_boxes_gpu(cur_points[:, 1:4].unsqueeze(0), bboxes[:, :7].unsqueeze(0)).squeeze()
            index = bboxes[:, -1] > 0
            cur_gt_corners = get_corner_with_target_v3(index.sum().item(), bboxes.cpu().numpy(), cur_points.cpu().numpy(), points_in_bboxes_index.cpu().numpy(),
                        gt_corners[b].cpu().numpy(), box_corners, b, self.use_corner0)
            gt_corners[b] = torch.tensor(cur_gt_corners, device=gt_corners.device)
        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}

        heatmap = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        gt_ind = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, encode_size), dtype = torch.float32).to(gt_boxes.device)

        center_ops_cuda.draw_corner_gpu_v1(gt_corners, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        offset = 0
        for task_id, task in enumerate(self.tasks):
            end = offset + len(task.class_names)
            heatmap_of_task = heatmap[:, offset:end, :, :]
            gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, encode_size)
            gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_encoding_merged = torch.zeros((batch_size, max_objs, encode_size), dtype=torch.float32).to(gt_boxes.device)
            offset = end
            for i in range(batch_size):
                mask = gt_mask_of_task[i] == 1
                mask_range = mask.sum().item()
                assert mask_range <= max_objs
                gt_mask_merged[i, :mask_range] = gt_mask_of_task[i, mask]
                gt_ind_merged[i, :mask_range] = gt_ind_of_task[i, mask]
                gt_cat_merged[i, :mask_range] = gt_cat_of_task[i, mask]
                gt_box_encoding_merged[i, :mask_range, :] = gt_box_encoding_of_task[i, mask, :]
                # only perform log on valid gt_box_encoding
                if not self.no_log:
                    gt_box_encoding_merged[i, :mask_range, 3:6] = torch.log(gt_box_encoding_merged[i, :mask_range, 3:6]) # log(wlh)

            heatmaps[task_id] = heatmap_of_task
            gt_inds[task_id] = gt_ind_merged.long()
            gt_masks[task_id] = gt_mask_merged.bool()
            gt_cats[task_id] = gt_cat_merged.long()
            gt_box_encodings[task_id] = gt_box_encoding_merged

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict

@numba.jit(nopython=True)
def get_corner(boxes_num, bboxes, cur_points, points_in_bboxes_index, gt_corners, box_corners, b,  use_corner0):
    for i in range(boxes_num):
        if use_corner0[int(bboxes[i, -1]-1)]:
            gt_corners[i, :] = box_corners[i, 0, :2]
        else:
            if (points_in_bboxes_index == i).sum() == 0:
                gt_corners[i, :] = box_corners[i, 0, :2]
                continue
            else:
                points_in_cur_bbox = cur_points[points_in_bboxes_index==i]
                cur_bbox_rotation_angle = -bboxes[i, 6]
                rotation_matrix = np.array([[np.cos(cur_bbox_rotation_angle), np.sin(cur_bbox_rotation_angle)],
                                            [-np.sin(cur_bbox_rotation_angle), np.cos(cur_bbox_rotation_angle)]])
                points_in_cur_bbox = points_in_cur_bbox[:, 1:3]
                center_in_velo = bboxes[i, :2]
                points_in_car_coor = points_in_cur_bbox @ rotation_matrix
                center_in_car_coor = center_in_velo @ rotation_matrix
                # points_in_car_coor, center_in_car_coor = \
                #     velo_to_car_cpu(points_in_cur_bbox[:, 1:3], bboxes[i, :2], -cur_bbox_rotation_angle)
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
                gt_corners[i, :2] = box_corners[i, max_1, :2]
                # gt_corners[i, :2] = (box_corners[i, max_1, :2] + bboxes[i, :2]) / 2.
    return gt_corners

@numba.jit(nopython=True)
def get_cornerV1(boxes_num, bboxes, cur_points, points_in_bboxes_index, gt_corners, box_corners, b,  use_corner0):
    index = np.array([[1, 3], [0, 2], [3, 1], [2, 0]])
    for i in range(boxes_num):
        if use_corner0[int(bboxes[i, -1]-1)]:
            gt_corners[i*3, :] = box_corners[i, 0, :2]
            gt_corners[i*3+1, :] = box_corners[i, 0, :2]
            gt_corners[i*3+2, :] = box_corners[i, 0, :2]
        else:
            if (points_in_bboxes_index == i).sum() == 0:
                gt_corners[i*3, :] = box_corners[i, 0, :2]
                gt_corners[i*3+1, :] = box_corners[i, 1, :2]
                gt_corners[i*3+2, :] = box_corners[i, 3, :2]
                continue
            else:
                points_in_cur_bbox = cur_points[points_in_bboxes_index==i]
                cur_bbox_rotation_angle = -bboxes[i, 6]
                rotation_matrix = np.array([[np.cos(cur_bbox_rotation_angle), np.sin(cur_bbox_rotation_angle)],
                                            [-np.sin(cur_bbox_rotation_angle), np.cos(cur_bbox_rotation_angle)]])
                points_in_cur_bbox = points_in_cur_bbox[:, 1:3]
                center_in_velo = bboxes[i, :2]
                points_in_car_coor = points_in_cur_bbox @ rotation_matrix
                center_in_car_coor = center_in_velo @ rotation_matrix
                # points_in_car_coor, center_in_car_coor = \
                #     velo_to_car_cpu(points_in_cur_bbox[:, 1:3], bboxes[i, :2], -cur_bbox_rotation_angle)
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
                gt_corners[i*3, :2] = box_corners[i, max_1, :2]
                gt_corners[i*3+1, :2] = box_corners[i, index[max_1][0], :2]
                gt_corners[i*3+2, :2] = box_corners[i, index[max_1][1], :2]
                # gt_corners[i, :2] = (box_corners[i, max_1, :2] + bboxes[i, :2]) / 2.
    return gt_corners

@numba.jit(nopython=True)
def get_corner_with_target(boxes_num, bboxes, cur_points, points_in_bboxes_index, gt_corners, box_corners, b,  use_corner0):
    for i in range(boxes_num):
        if use_corner0[int(bboxes[i, -1]-1)]:
            gt_corners[i, :2] = box_corners[i, 0, :2]
            gt_corners[i, 2] = bboxes[i, 2]
            gt_corners[i, 3:6] = bboxes[i, 3:6]
            gt_corners[i, 6] = bboxes[i, 6]
            diag_dist = np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2.
            gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
            gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
            gt_corners[i, 9] = bboxes[i, 7]
        else:
            if (points_in_bboxes_index == i).sum() == 0:
                gt_corners[i, :2] = box_corners[i, 0, :2]
                gt_corners[i, 2] = bboxes[i, 2]
                gt_corners[i, 3:6] = bboxes[i, 3:6]
                gt_corners[i, 6] = bboxes[i, 6]
                diag_dist = np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2.
                gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
                gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
                gt_corners[i, 9] = bboxes[i, 7]
                continue
            else:
                points_in_cur_bbox = cur_points[points_in_bboxes_index==i]
                cur_bbox_rotation_angle = -bboxes[i, 6]
                rotation_matrix = np.array([[np.cos(cur_bbox_rotation_angle), np.sin(cur_bbox_rotation_angle)],
                                            [-np.sin(cur_bbox_rotation_angle), np.cos(cur_bbox_rotation_angle)]])
                points_in_cur_bbox = points_in_cur_bbox[:, 1:3]
                center_in_velo = bboxes[i, :2]
                points_in_car_coor = points_in_cur_bbox @ rotation_matrix
                center_in_car_coor = center_in_velo @ rotation_matrix
                # points_in_car_coor, center_in_car_coor = \
                #     velo_to_car_cpu(points_in_cur_bbox[:, 1:3], bboxes[i, :2], -cur_bbox_rotation_angle)
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
                gt_corners[i, :2] = box_corners[i, max_1, :2]
                gt_corners[i, 2] = bboxes[i, 2]
                gt_corners[i, 3:6] = bboxes[i, 3:6]
                gt_corners[i, 6] = bboxes[i, 6]
                diag_dist = np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2.
                gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
                gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
                gt_corners[i, 9] = bboxes[i, 7]
    return gt_corners

@numba.jit(nopython=True)
def get_corner_with_target_v2(boxes_num, bboxes, cur_points, points_in_bboxes_index, gt_corners, box_corners, b,  use_corner0):
    for i in range(boxes_num):
        if use_corner0[int(bboxes[i, -1]-1)]:
            gt_corners[i, :2] = box_corners[i, 0, :2]
            gt_corners[i, 2] = bboxes[i, 2]
            gt_corners[i, 3:6] = bboxes[i, 3:6]
            gt_corners[i, 6] = bboxes[i, 6]
            diag_dist = (np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2. + 1e-4)
            gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
            gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
            gt_corners[i, 9] = bboxes[i, 7]
        else:
            if (points_in_bboxes_index == i).sum() == 0:
                gt_corners[i, :2] = box_corners[i, 0, :2]
                gt_corners[i, 2] = bboxes[i, 2]
                gt_corners[i, 3:6] = bboxes[i, 3:6]
                gt_corners[i, 6] = bboxes[i, 6]
                diag_dist = (np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2. + 1e-4)
                gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
                gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
                gt_corners[i, 9] = bboxes[i, 7]
                continue
            else:
                points_in_cur_bbox = cur_points[points_in_bboxes_index==i]
                cur_bbox_rotation_angle = -bboxes[i, 6]
                rotation_matrix = np.array([[np.cos(cur_bbox_rotation_angle), np.sin(cur_bbox_rotation_angle)],
                                            [-np.sin(cur_bbox_rotation_angle), np.cos(cur_bbox_rotation_angle)]])
                points_in_cur_bbox = points_in_cur_bbox[:, 1:3]
                center_in_velo = bboxes[i, :2]
                points_in_car_coor = points_in_cur_bbox @ rotation_matrix
                center_in_car_coor = center_in_velo @ rotation_matrix
                # points_in_car_coor, center_in_car_coor = \
                #     velo_to_car_cpu(points_in_cur_bbox[:, 1:3], bboxes[i, :2], -cur_bbox_rotation_angle)
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
                gt_corners[i, :2] = box_corners[i, max_1, :2]
                gt_corners[i, 2] = bboxes[i, 2]
                gt_corners[i, 3:6] = bboxes[i, 3:6]
                gt_corners[i, 6] = bboxes[i, 6]
                diag_dist = np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2. + 1e-4
                gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
                gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
                gt_corners[i, 9] = bboxes[i, 7]
    return gt_corners

@numba.jit(nopython=True)
def get_corner_with_target_v3(boxes_num, bboxes, cur_points, points_in_bboxes_index, gt_corners, box_corners, b,  use_corner0):
    for i in range(boxes_num):
        if use_corner0[int(bboxes[i, -1]-1)]:
            gt_corners[i, :2] = bboxes[i, :2]
            gt_corners[i, 2] = bboxes[i, 2]
            gt_corners[i, 3:6] = bboxes[i, 3:6]
            gt_corners[i, 6] = bboxes[i, 6]
            # diag_dist = (np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2. + 1e-4)
            gt_corners[i, 7] = 0.
            gt_corners[i, 8] = 0.
            gt_corners[i, 9] = bboxes[i, 7]
        else:
            if (points_in_bboxes_index == i).sum() == 0:
                gt_corners[i, :2] = box_corners[i, 0, :2]
                gt_corners[i, 2] = bboxes[i, 2]
                gt_corners[i, 3:6] = bboxes[i, 3:6]
                gt_corners[i, 6] = bboxes[i, 6]
                diag_dist = (np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2. + 1e-4)
                gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
                gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
                gt_corners[i, 9] = bboxes[i, 7]
                continue
            else:
                points_in_cur_bbox = cur_points[points_in_bboxes_index==i]
                cur_bbox_rotation_angle = -bboxes[i, 6]
                rotation_matrix = np.array([[np.cos(cur_bbox_rotation_angle), np.sin(cur_bbox_rotation_angle)],
                                            [-np.sin(cur_bbox_rotation_angle), np.cos(cur_bbox_rotation_angle)]])
                points_in_cur_bbox = points_in_cur_bbox[:, 1:3]
                center_in_velo = bboxes[i, :2]
                points_in_car_coor = points_in_cur_bbox @ rotation_matrix
                center_in_car_coor = center_in_velo @ rotation_matrix
                # points_in_car_coor, center_in_car_coor = \
                #     velo_to_car_cpu(points_in_cur_bbox[:, 1:3], bboxes[i, :2], -cur_bbox_rotation_angle)
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
                gt_corners[i, :2] = box_corners[i, max_1, :2]
                gt_corners[i, 2] = bboxes[i, 2]
                gt_corners[i, 3:6] = bboxes[i, 3:6]
                gt_corners[i, 6] = bboxes[i, 6]
                diag_dist = np.sqrt(bboxes[i, 3]**2 + bboxes[i, 4]**2) / 2. + 1e-4
                gt_corners[i, 7] = (bboxes[i, 0] - gt_corners[i, 0]) / diag_dist
                gt_corners[i, 8] = (bboxes[i, 1] - gt_corners[i, 1]) / diag_dist
                gt_corners[i, 9] = bboxes[i, 7]
    return gt_corners
