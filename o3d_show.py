import open3d as o3d
import numpy as np
import torch


box_colormap = [
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0]
]

def show_result(vis, points, gt_boxes=None, gt_labels=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    pcobj = o3d.geometry.PointCloud()
    if not isinstance(points, np.ndarray):
        points = points.data.cpu().numpy()
    points[:, 1] = -points[:, 1]
    pcobj.points = o3d.utility.Vector3dVector(points[:, :3])
    points_colors = np.tile(np.array([0.5, 0.5, 0.5]), (points.shape[0], 1))
    pcobj.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcobj)
    if gt_boxes is not None:
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.data.cpu().numpy()
            gt_labels = gt_labels.data.cpu().numpy()
        vis = draw_bboxes(vis, gt_boxes, gt_labels, color=(0, 1, 0))
    if ref_boxes is not None:
        if isinstance(ref_boxes, torch.Tensor):
            ref_boxes = ref_boxes.data.cpu().numpy()
            ref_labels = ref_labels.data.cpu().numpy()
        vis = draw_bboxes(vis, ref_boxes, ref_labels)
    return vis

def draw_bboxes(vis, bboxes, labels, color=None):
    for i in range(len(bboxes)):
        center = bboxes[i, :3]
        center[1] = -center[1]
        dim = bboxes[i, 3:6]
        yaw = np.zeros(3)
        yaw[2] = -bboxes[i, 6]
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        if color is None:
            line_set.paint_uniform_color(tuple(box_colormap[labels[i]-1]))
        else:
            line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)
    return vis