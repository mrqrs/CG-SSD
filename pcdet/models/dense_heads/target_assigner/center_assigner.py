import torch
import math

from zmq import device
from ....ops.center_ops import center_ops_cuda
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_bev_gpu, points_in_boxes_gpu
from pcdet.utils.box_utils import boxes_to_corners_3d
import numba
import numpy as np
# import matplotlib.pyplot as plt


class CenterAssigner(object):
    def __init__(self, assigner_cfg, num_classes, no_log, grid_size, pc_range, voxel_size):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.num_classes = num_classes
        self.tasks = assigner_cfg.tasks
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.class_to_idx = assigner_cfg.mapping
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log
        self.assigner_cfg = assigner_cfg

    def gaussian_radius(self, height, width, min_overlap=0.5):
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def gaussian_2d(self, shape, sigma = 1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        mesh_m = torch.arange(start=-m, end=m+1, step=1, dtype=torch.float32)
        mesh_n = torch.arange(start=-n, end=n+1, step=1, dtype=torch.float32)
        y, x = torch.meshgrid([mesh_m, mesh_n])
        h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
        eps = 1e-7
        h[h < eps * h.max()] = 0
        return h

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)
        gaussian = gaussian.to(heatmap.device)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            heatmap[y - top:y + bottom, x - left:x + right] = torch.stack([masked_heatmap, masked_gaussian * k], dim = 0).max(0)[0]
        return heatmap

    def limit_period(self, val, offset=0.5, period=math.pi):
        return val - math.floor(val / period + offset) * period

    def assign_targets_v1(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)

        Returns:

        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor # grid_size WxHxD feature_map_size WxH

        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, -1] #begin from 1
        gt_boxes = gt_boxes[:, :, :-1]

        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}
        for task_id, task in enumerate(self.tasks):
            heatmaps[task_id] = []
            gt_inds[task_id] = []
            gt_masks[task_id] = []
            gt_box_encodings[task_id] = []
            gt_cats[task_id] = []

        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            for task_id, task in enumerate(self.tasks):
                # heatmap size is supposed to be (cls_group, H, W)
                heatmap = torch.zeros((len(task.class_names), feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(cur_gt.device) #transpose ????
                gt_ind = torch.zeros(max_objs, dtype=torch.long).to(cur_gt.device)
                gt_mask = torch.zeros(max_objs, dtype=torch.bool).to(cur_gt.device)
                gt_cat = torch.zeros(max_objs, dtype=torch.long).to(cur_gt.device)
                gt_box_encoding = torch.zeros((max_objs, 10), dtype=torch.float32).to(cur_gt.device)

                cur_gts_of_task = []
                cur_classes_of_task = []
                class_offset = 0
                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_mask = (cur_gt_classes == class_idx)
                    cur_gt_of_task = cur_gt[class_mask]
                    cur_class_of_task = cur_gt.new_full((cur_gt_of_task.shape[0],), class_offset).long()
                    cur_gts_of_task.append(cur_gt_of_task)
                    cur_classes_of_task.append(cur_class_of_task)
                    class_offset += 1
                cur_gts_of_task = torch.cat(cur_gts_of_task, dim = 0)
                cur_classes_of_task = torch.cat(cur_classes_of_task, dim = 0)

                num_boxes_of_task = cur_gts_of_task.shape[0]
                for i in range(num_boxes_of_task):
                    cat = cur_classes_of_task[i]
                    x, y, z, w, l, h, r, vx, vy = cur_gts_of_task[i]
                    #r -> [-pi, pi]
                    r = self.limit_period(r, offset=0.5, period=math.pi * 2)
                    w, l = w / self.voxel_size[0] / self.out_size_factor, l / self.voxel_size[1] / self.out_size_factor
                    radius = self.gaussian_radius(l, w, min_overlap=self.gaussian_overlap)
                    radius = max(self._min_radius, int(radius))

                    coor_x = (x - self.pc_range[0]) / self.voxel_size[0] / self.out_size_factor
                    coor_y = (y - self.pc_range[1]) / self.voxel_size[1] / self.out_size_factor
                    ct_ft = torch.tensor([coor_x, coor_y], dtype = torch.float32)
                    ct_int = ct_ft.int() #float to int conversion torch/np

                    if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                        continue

                    self.draw_gaussian(heatmap[cat], ct_int, radius) #pass functions

                    gt_cat[i] = cat
                    gt_mask[i] = 1
                    gt_ind[i] = ct_int[1] * feature_map_size[0] + ct_int[0]
                    assert gt_ind[i] < feature_map_size[0] * feature_map_size[1]
                    # Note that w,l has been modified, so in box encoding we use original w,l,h
                    if not self.no_log:
                        gt_box_encoding[i] = torch.tensor([ct_ft[0] - ct_int[0],
                                                           ct_ft[1] - ct_int[1],
                                                           z,
                                                           math.log(cur_gts_of_task[i,3]),
                                                           math.log(cur_gts_of_task[i,4]),
                                                           math.log(cur_gts_of_task[i,5]),
                                                           math.sin(r),
                                                           math.cos(r),
                                                           vx,
                                                           vy
                                                           ], dtype=torch.float32).to(gt_box_encoding.device)
                    else:
                        gt_box_encoding[i] = torch.tensor([ct_ft[0] - ct_int[0],
                                                           ct_ft[1] - ct_int[1],
                                                           z,
                                                           cur_gts_of_task[i,3],
                                                           cur_gts_of_task[i,4],
                                                           cur_gts_of_task[i,5],
                                                           math.sin(r),
                                                           math.cos(r),
                                                           vx,
                                                           vy
                                                           ], dtype=torch.float32).to(gt_box_encoding.device)

                heatmaps[task_id].append(heatmap)
                gt_inds[task_id].append(gt_ind)
                gt_cats[task_id].append(gt_cat)
                gt_masks[task_id].append(gt_mask)
                gt_box_encodings[task_id].append(gt_box_encoding)

        for task_id, tasks in enumerate(self.tasks):
            heatmaps[task_id] = torch.stack(heatmaps[task_id], dim = 0).contiguous()
            gt_inds[task_id] = torch.stack(gt_inds[task_id], dim = 0).contiguous()
            gt_masks[task_id] = torch.stack(gt_masks[task_id], dim = 0).contiguous()
            gt_cats[task_id] = torch.stack(gt_cats[task_id], dim = 0).contiguous()
            gt_box_encodings[task_id] = torch.stack(gt_box_encodings[task_id], dim = 0).contiguous()

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict

    def assign_targets_v2(self, gt_boxes):
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
        gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, code_size), dtype = torch.float32).to(gt_boxes.device)

        center_ops_cuda.draw_center_gpu(gt_boxes, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        offset = 0
        for task_id, task in enumerate(self.tasks):
            end = offset + len(task.class_names)
            heatmap_of_task = heatmap[:, offset:end, :, :]
            gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, code_size)
            gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_encoding_merged = torch.zeros((batch_size, max_objs, code_size), dtype=torch.float32).to(gt_boxes.device)
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

class CenterAssigner_Aux(object):
    def __init__(self, assigner_cfg, num_classes, no_log, grid_size, pc_range, voxel_size):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.num_classes = num_classes
        self.tasks = assigner_cfg.tasks
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.class_to_idx = assigner_cfg.mapping
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log

    def gaussian_radius(self, height, width, min_overlap=0.5):
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def gaussian_2d(self, shape, sigma = 1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        mesh_m = torch.arange(start=-m, end=m+1, step=1, dtype=torch.float32)
        mesh_n = torch.arange(start=-n, end=n+1, step=1, dtype=torch.float32)
        y, x = torch.meshgrid([mesh_m, mesh_n])
        h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
        eps = 1e-7
        h[h < eps * h.max()] = 0
        return h

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)
        gaussian = gaussian.to(heatmap.device)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            heatmap[y - top:y + bottom, x - left:x + right] = torch.stack([masked_heatmap, masked_gaussian * k], dim = 0).max(0)[0]
        return heatmap

    def limit_period(self, val, offset=0.5, period=math.pi):
        return val - math.floor(val / period + offset) * period

    def assign_targets_3p(self, gt_boxes, points, use_corner0):
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

        gt_corners0 = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 5), device=gt_boxes.device)
        gt_corners1 = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 5), device=gt_boxes.device)
        gt_corners2 = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 5), device=gt_boxes.device)
        
        for b in range(batch_size):
            bboxes = gt_boxes[b]
            box_corners = boxes_to_corners_3d(bboxes[:, :7].data.cpu().numpy())
            box_corners = box_corners[:, [0, 1, 2, 3], :]
            cur_points = points[points[:, 0]==b]
            points_in_bboxes_index = points_in_boxes_gpu(cur_points[:, 1:4].unsqueeze(0), bboxes[:, :7].unsqueeze(0)).squeeze()
            index = bboxes[:, -1] > 0
            corner_index = np.array([[2, 3, 1], [3, 2, 0], [0, 1, 3], [1, 0, 2]]) ###flw
           
            cur_gt_corners0, cur_gt_corners1, cur_gt_corners2 = get_corner_aux_3p(index.sum().item(), bboxes.cpu().numpy(), cur_points.cpu().numpy(), points_in_bboxes_index.cpu().numpy(),
                        gt_corners0[b].cpu().numpy(), gt_corners1[b].cpu().numpy(), gt_corners2[b].cpu().numpy(), corner_index, box_corners, b,use_corner0)
            gt_corners0[b] = torch.tensor(cur_gt_corners0, device=gt_corners0.device)
            gt_corners1[b] = torch.tensor(cur_gt_corners1, device=gt_corners0.device)
            gt_corners2[b] = torch.tensor(cur_gt_corners2, device=gt_corners0.device)


        heatmaps0 = {}
        gt_inds0 = {}
        gt_masks0 = {}
        gt_box_offset0_encodings = {}
        gt_cats0 = {}

        heatmaps1 = {}
        gt_inds1 = {}
        gt_masks1 = {}
        gt_box_offset1_encodings = {}
        gt_cats1 = {}

        heatmaps2 = {}
        gt_inds2 = {}
        gt_masks2 = {}
        gt_box_offset2_encodings = {}
        gt_cats2 = {}

        # heatmap = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        heatmap0 = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        heatmap1 = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        heatmap2 = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)

        # gt_ind = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_ind0 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_ind1 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_ind2 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)

        # gt_mask = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask0 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask1 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask2 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)

        # gt_cat = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat0 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat1 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat2 = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)

        # gt_cnt = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt0 = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt1 = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt2 = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)

        # gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, code_size), dtype = torch.float32).to(gt_boxes.device)
        gt_box_offset0_encoding = torch.zeros((batch_size, num_classes, max_objs, 2), dtype = torch.float32).to(gt_boxes.device)
        gt_box_offset1_encoding = torch.zeros((batch_size, num_classes, max_objs, 2), dtype = torch.float32).to(gt_boxes.device)
        gt_box_offset2_encoding = torch.zeros((batch_size, num_classes, max_objs, 2), dtype = torch.float32).to(gt_boxes.device)

        # center_ops_cuda.draw_center_gpu(gt_boxes, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt,
        #                 self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
        #                 self.out_size_factor, self.gaussian_overlap)
        
        center_ops_cuda.draw_aux_gpu(gt_corners0, heatmap0, gt_ind0, gt_mask0, gt_cat0, gt_box_offset0_encoding, gt_cnt0,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        center_ops_cuda.draw_aux_gpu(gt_corners1, heatmap1, gt_ind1, gt_mask1, gt_cat1, gt_box_offset1_encoding, gt_cnt1,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)
        
        center_ops_cuda.draw_aux_gpu(gt_corners2, heatmap2, gt_ind2, gt_mask2, gt_cat2, gt_box_offset2_encoding, gt_cnt2,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        offset = 0
        for task_id, task in enumerate(self.tasks):
            end = offset + len(task.class_names)
            # heatmap_of_task = heatmap[:, offset:end, :, :]
            # gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            # gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            # gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            # gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, code_size)
            # gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            # gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            # gt_cat_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            # gt_box_encoding_merged = torch.zeros((batch_size, max_objs, code_size), dtype=torch.float32).to(gt_boxes.device)

            heatmap0_of_task = heatmap0[:, offset:end, :, :]
            gt_ind0_of_task = gt_ind0[:, offset:end, :].reshape(batch_size, -1)
            gt_mask0_of_task = gt_mask0[:, offset:end, :].reshape(batch_size, -1)
            gt_cat0_of_task = gt_cat0[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_offset0_encoding_of_task = gt_box_offset0_encoding[:, offset:end, :, :].reshape(batch_size, -1, 2)
            gt_ind0_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask0_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat0_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_offset0_encoding_merged = torch.zeros((batch_size, max_objs, 2), dtype=torch.float32).to(gt_boxes.device)

            heatmap1_of_task = heatmap1[:, offset:end, :, :]
            gt_ind1_of_task = gt_ind1[:, offset:end, :].reshape(batch_size, -1)
            gt_mask1_of_task = gt_mask1[:, offset:end, :].reshape(batch_size, -1)
            gt_cat1_of_task = gt_cat1[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_offset1_encoding_of_task = gt_box_offset1_encoding[:, offset:end, :, :].reshape(batch_size, -1, 2)
            gt_ind1_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask1_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat1_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_offset1_encoding_merged = torch.zeros((batch_size, max_objs, 2), dtype=torch.float32).to(gt_boxes.device)

            heatmap2_of_task = heatmap2[:, offset:end, :, :]
            gt_ind2_of_task = gt_ind2[:, offset:end, :].reshape(batch_size, -1)
            gt_mask2_of_task = gt_mask2[:, offset:end, :].reshape(batch_size, -1)
            gt_cat2_of_task = gt_cat2[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_offset2_encoding_of_task = gt_box_offset2_encoding[:, offset:end, :, :].reshape(batch_size, -1, 2)
            gt_ind2_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask2_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat2_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_offset2_encoding_merged = torch.zeros((batch_size, max_objs, 2), dtype=torch.float32).to(gt_boxes.device)


            offset = end
            for i in range(batch_size):
                # mask = gt_mask_of_task[i] == 1
                # mask_range = mask.sum().item()
                # assert mask_range <= max_objs
                # gt_mask_merged[i, :mask_range] = gt_mask_of_task[i, mask]
                # gt_ind_merged[i, :mask_range] = gt_ind_of_task[i, mask]
                # gt_cat_merged[i, :mask_range] = gt_cat_of_task[i, mask]
                # gt_box_encoding_merged[i, :mask_range, :] = gt_box_encoding_of_task[i, mask, :]
                # # only perform log on valid gt_box_encoding
                # if not self.no_log:
                #     gt_box_encoding_merged[i, :mask_range, 3:6] = torch.log(gt_box_encoding_merged[i, :mask_range, 3:6]) # log(wlh)
                
                mask0 = gt_mask0_of_task[i] == 1
                mask0_range = mask0.sum().item()
                assert mask0_range <= max_objs
                gt_mask0_merged[i, :mask0_range] = gt_mask0_of_task[i, mask0]
                gt_ind0_merged[i, :mask0_range] = gt_ind0_of_task[i, mask0]
                gt_cat0_merged[i, :mask0_range] = gt_cat0_of_task[i, mask0]
                gt_box_offset0_encoding_merged[i, :mask0_range, :] = gt_box_offset0_encoding_of_task[i, mask0, :]

                mask1 = gt_mask1_of_task[i] == 1
                mask1_range = mask1.sum().item()
                assert mask1_range <= max_objs
                gt_mask1_merged[i, :mask1_range] = gt_mask1_of_task[i, mask1]
                gt_ind1_merged[i, :mask1_range] = gt_ind1_of_task[i, mask1]
                gt_cat1_merged[i, :mask1_range] = gt_cat1_of_task[i, mask1]
                gt_box_offset1_encoding_merged[i, :mask1_range, :] = gt_box_offset1_encoding_of_task[i, mask1, :]

                mask2 = gt_mask2_of_task[i] == 1
                mask2_range = mask2.sum().item()
                assert mask2_range <= max_objs
                gt_mask2_merged[i, :mask2_range] = gt_mask2_of_task[i, mask2]
                gt_ind2_merged[i, :mask2_range] = gt_ind2_of_task[i, mask2]
                gt_cat2_merged[i, :mask2_range] = gt_cat2_of_task[i, mask2]
                gt_box_offset2_encoding_merged[i, :mask2_range, :] = gt_box_offset2_encoding_of_task[i, mask2, :]

            # heatmaps[task_id] = heatmap_of_task
            # gt_inds[task_id] = gt_ind_merged.long()
            # gt_masks[task_id] = gt_mask_merged.bool()
            # gt_cats[task_id] = gt_cat_merged.long()
            # gt_box_encodings[task_id] = gt_box_encoding_merged

            heatmaps0[task_id] = heatmap0_of_task
            gt_inds0[task_id] = gt_ind0_merged.long()
            gt_masks0[task_id] = gt_mask0_merged.bool()
            gt_cats0[task_id] = gt_cat0_merged.long()
            gt_box_offset0_encodings[task_id] = gt_box_offset0_encoding_merged

            heatmaps1[task_id] = heatmap1_of_task
            gt_inds1[task_id] = gt_ind1_merged.long()
            gt_masks1[task_id] = gt_mask1_merged.bool()
            gt_cats1[task_id] = gt_cat1_merged.long()
            gt_box_offset1_encodings[task_id] = gt_box_offset1_encoding_merged

            heatmaps2[task_id] = heatmap2_of_task
            gt_inds2[task_id] = gt_ind2_merged.long()
            gt_masks2[task_id] = gt_mask2_merged.bool()
            gt_cats2[task_id] = gt_cat2_merged.long()
            gt_box_offset2_encodings[task_id] = gt_box_offset2_encoding_merged

        target_dict = {
            # 'heatmap': heatmaps,
            # 'ind': gt_inds,
            # 'mask': gt_masks,
            # 'cat': gt_cats,
            # 'box_encoding': gt_box_encodings,

            'heatmap0': heatmaps0,
            'ind0': gt_inds0,
            'mask0': gt_masks0,
            'cat0': gt_cats0,
            'offset0': gt_box_offset0_encodings,

            'heatmap1': heatmaps1,
            'ind1': gt_inds1,
            'mask1': gt_masks1,
            'cat1': gt_cats1,
            'offset1': gt_box_offset1_encodings,

            'heatmap2': heatmaps2,
            'ind2': gt_inds2,
            'mask2': gt_masks2,
            'cat2': gt_cats2,
            'offset2': gt_box_offset2_encodings
        }
        return target_dict


@numba.jit(nopython=True)
def get_corner_aux_3p(boxes_num, bboxes, cur_points, points_in_bboxes_index, gt_corners0, gt_corners1, gt_corners2, corner_index, box_corners, b,  use_corner0):
    for i in range(boxes_num):
        if use_corner0[int(bboxes[i, -1]-1)]:
            gt_corners0[i, :2] = box_corners[i, 0, :2]
            gt_corners0[i, 2:4] = bboxes[i, 3:5]
            gt_corners0[i, 4] = bboxes[i, -1]
            gt_corners1[i, :2] = box_corners[i, 1, :2]
            gt_corners1[i, 2:4] = bboxes[i, 3:5]
            gt_corners1[i, 4] = bboxes[i, -1]
            gt_corners2[i, :2] = box_corners[i, 3, :2]
            gt_corners2[i, 2:4] = bboxes[i, 3:5]
            gt_corners2[i, 4] = bboxes[i, -1]
        else:
            if (points_in_bboxes_index == i).sum() == 0:
                gt_corners0[i, :2] = box_corners[i, 0, :2]
                gt_corners0[i, 2:4] = bboxes[i, 3:5]
                gt_corners0[i, 4] = bboxes[i, -1]
                gt_corners1[i, :2] = box_corners[i, 1, :2]
                gt_corners1[i, 2:4] = bboxes[i, 3:5]
                gt_corners1[i, 4] = bboxes[i, -1]
                gt_corners2[i, :2] = box_corners[i, 3, :2]
                gt_corners2[i, 2:4] = bboxes[i, 3:5]
                gt_corners2[i, 4] = bboxes[i, -1]
                continue
            else:
                points_in_cur_bbox = cur_points[points_in_bboxes_index==i]
                cur_bbox_rotation_angle = -bboxes[i, 6]
                rotation_matrix = np.array([[np.cos(cur_bbox_rotation_angle), np.sin(cur_bbox_rotation_angle)],
                                            [-np.sin(cur_bbox_rotation_angle), np.cos(cur_bbox_rotation_angle)]])
                points_in_cur_bbox = np.ascontiguousarray(points_in_cur_bbox[:, 1:3])
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
                gt_corners0[i, :2] = box_corners[i, corner_index[max_1][0], :2]
                gt_corners0[i, 2] = bboxes[i, 3]
                gt_corners0[i, 3] = bboxes[i, 4]
                gt_corners0[i, 4] = bboxes[i, -1]
                gt_corners1[i, :2] = box_corners[i, corner_index[max_1][1], :2]
                gt_corners1[i, 2] = bboxes[i, 3]
                gt_corners1[i, 3] = bboxes[i, 4]
                gt_corners1[i, 4] = bboxes[i, -1]
                gt_corners2[i, :2] = box_corners[i, corner_index[max_1][2], :2]
                gt_corners2[i, 2] = bboxes[i, 3]
                gt_corners2[i, 3] = bboxes[i, 4]
                gt_corners2[i, 4] = bboxes[i, -1]

    return gt_corners0, gt_corners1, gt_corners2
             