import numpy as np
from numpy.core.numeric import NaN
import torch


def corners_in_image_index(vis_corners, pc_range, voxel_size, out_size_factor, bbox, reg_target):
    """
    :param vis_corners: corners used to calculate the spots of heatmap
    :param pc_range: used to calculate the index of spots
    :param voxel_size: used to calculate the index of spots
    :param out_size_factor: used to calculate the index of spots
    :return:
        corner_index: the corner index of heat spot
    """

    x_index = torch.floor((vis_corners[:, 0] - pc_range[0]) / (voxel_size[0]*out_size_factor))
    y_index = torch.floor((vis_corners[:, 1] - pc_range[1]) / (voxel_size[1]*out_size_factor))
    index = torch.cat((x_index.reshape((-1, 1)), y_index.reshape((-1, 1))), dim=1)
    reg_target[:, 0] = vis_corners[:, 0] - pc_range[0] - (x_index.type(torch.int)) * voxel_size[0]*out_size_factor
    reg_target[:, 1] = vis_corners[:, 1] - pc_range[1] - (y_index.type(torch.int)) * voxel_size[1]*out_size_factor
    return index, reg_target

def create_heatmap(bbox, corners_in_image_index, heatmap, reg_target, reg_target_map, voxel_size,
                   out_size_factor, min_radius, min_overlap, gt_labels, cls_label):
    for j in range(heatmap.shape[0]):
        cur_heatmap = heatmap[j]
        cur_selected = gt_labels == cls_label[j]
        cur_bbox = bbox[cur_selected]
        cur_corners_in_image_index = corners_in_image_index[cur_selected]
        for i in range(cur_bbox.shape[0]):
            out_voxel_size = torch.tensor(voxel_size[:2], device=heatmap.device) * out_size_factor
            l, w = cur_bbox[i, [3, 4]] / out_voxel_size
            ct = cur_corners_in_image_index[i].type(torch.float32)
            radius = gaussian_radius((l, w), min_overlap)
            radius = int(max(min_radius, radius))
            x, y = int(ct[0]), int(ct[1])
            height, width = cur_heatmap.shape[0:2]
            left, right = min(x, radius), min(width - x, radius + 1)
            top, bottom = min(y, radius), min(height - y, radius + 1)
            reg_target_map[y - top:y + bottom, x - left:x + right, :] = reg_target[i]
            draw_umich_gaussian(cur_heatmap, ct, radius)
        heatmap[j] = cur_heatmap
    return heatmap, reg_target_map

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(gaussian[radius - top:radius + bottom, radius - left:radius + right]).to(heatmap.device, torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def velo_to_car_gpu(points, center, angle):
    device = points.device
    rotation_matrix = torch.tensor([[torch.cos(angle), torch.sin(angle)],
                                [-torch.sin(angle), torch.cos(angle)]], device=device)
    points_in_car_coor = points @ rotation_matrix
    # corners_in_car_coor = corners @ rotation_matrix
    center_in_car_coor = center @ rotation_matrix
    return points_in_car_coor, center_in_car_coor

def velo_to_car_cpu(points, center, angle):
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]])
    points_in_car_coor = points @ rotation_matrix
    # corners_in_car_coor = corners @ rotation_matrix
    center_in_car_coor = center @ rotation_matrix
    return points_in_car_coor, center_in_car_coor