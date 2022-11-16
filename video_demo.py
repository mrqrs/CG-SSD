import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils.o3d_show import show_result
from pcdet.datasets import build_dataloader
# from pcdet.datasets.kitti.kitti_dataset import KittiDataset
# from pcdet.ops.iou3d.iou3d_utils import boxes_iou3d_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import open3d as o3d






def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='./cfgs/once_models/sup_models/cornernet3d_v1.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/once/data/000027/lidar_roof/1616101913900.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str,
                        default='../output/cfgs/once_models/sup_models/cornernet3d_v1/default/ckpt/checkpoint_epoch_61.pth',
                        help='specify the pretrained model')
    parser.add_argument('--idx', type=int, default=211)
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=8, logger=logger, training=False
    )
    logger.info(f'Total number of samples: \t{len(test_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    idx = args.idx
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.show_coordinate_frame = True
    point3d = o3d.geometry.PointCloud()
    line_set = o3d.geometry.LineSet()
    vis.add_geometry(point3d)
    vis.add_geometry(line_set)
    for i in range(len(test_set)):
        with torch.no_grad():
            data_dict = test_set[i]
            # print(data_dict['frame_id'])
            # logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = test_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            gt_boxes = data_dict['gt_boxes'][0][:, :7]
            gt_labels = data_dict['gt_boxes'][0][:, -1].long()

            # iou3d_rcnn = boxes_iou3d_gpu(pred_dicts[0]['pred_boxes'], gt_boxes)
            # iou3d_rcnn = iou3d_rcnn.max(1)[0]

            vis = show_result(vis, points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, gt_labels=gt_labels,
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'])
            # vis.run()
            vis.poll_events()
            vis.update_renderer()
            vis.clear_geometries()
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
