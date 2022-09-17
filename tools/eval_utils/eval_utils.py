import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):

        # add temperature for adpative radius learning
        if cfg.OPTIMIZATION.get('USE_TEMPERATURE', False):
            batch_dict.update({'temperature': cfg.OPTIMIZATION.DECAY_TEMPERATURE[-1]})

        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    
    # ### test
    # frame_id = []
    # pred_data = pickle.load(open('/opt/data/private/result.pkl', 'rb'))
    # for i in range(len(pred_data)):
    #     frame_id.append(pred_data[i]['frame_id'])
    # test_submit = []
    # for j in range(len(det_annos)):
    #     if det_annos[j]['frame_id'] in frame_id:
    #         test_submit.append(det_annos[j])
    # with open(result_dir / 'result.pkl', 'wb') as f:
    #     pickle.dump(test_submit, f)
    
    ## val
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def cal_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    angle_dist_5055 = []
    angle_dist_5560 = []
    angle_dist_6065 = []
    angle_dist_6570 = []
    angle_dist_70 = []

    center_dist_5055 = []
    center_dist_5560 = []
    center_dist_6065 = []
    center_dist_6570 = []
    center_dist_70 = []

    w_dist_5055 = []
    w_dist_5560 = []
    w_dist_6065 = []
    w_dist_6570 = []
    w_dist_70 = []

    l_dist_5055 = []
    l_dist_5560 = []
    l_dist_6065 = []
    l_dist_6570 = []
    l_dist_70 = []

    h_dist_5055 = []
    h_dist_5560 = []
    h_dist_6065 = []
    h_dist_6570 = []
    h_dist_70 = []

    car = []
    bus = []
    truck = []
    car_iou = []
    bus_iou = []
    truck_iou = []

    car_number_5055, car_number_5560, car_number_6065, car_number_6570, car_number_70 = 0, 0, 0, 0, 0
    bus_number_5055, bus_number_5560, bus_number_6065, bus_number_6570, bus_number_70, = 0, 0, 0, 0, 0
    truck_number_5055, truck_number_5560, truck_number_6065, truck_number_6570, truck_number_70, = 0, 0, 0, 0, 0
    car_number, bus_number, truck_number = 0, 0, 0
    gt_car_number, gt_bus_number, gt_truck_number = 0, 0, 0

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):

        # add temperature for adpative radius learning
        if cfg.OPTIMIZATION.get('USE_TEMPERATURE', False):
            batch_dict.update({'temperature': cfg.OPTIMIZATION.DECAY_TEMPERATURE[-1]})

        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        
        gt_boxes = batch_dict['gt_boxes']
        gt_labels = gt_boxes[..., -1]
        for j in range(batch_dict['batch_size']):
            cur_batch_gt_boxes = gt_boxes[j]
            pred_boxes = pred_dicts[j]['pred_boxes']
            pred_labels = pred_dicts[j]['pred_labels']

            car_index = pred_labels == 1
            gt_car_index = cur_batch_gt_boxes[:, -1] == 1
            car_number += car_index.sum()
            gt_car_number += gt_car_index.sum()
            if car_index.sum() > 0:
                if gt_car_index.sum()==0:
                    fake_iou = torch.zeros(car_index.sum(), 1, device=gt_boxes.device)
                    car_iou.append(fake_iou)
                    fake_gt = torch.zeros_like(pred_boxes[car_index]).reshape(-1, 7)
                    car.append(torch.cat([pred_boxes[car_index], fake_gt], dim=1))
                else:
                    iou3d_rcnn = boxes_iou3d_gpu(pred_boxes[car_index], cur_batch_gt_boxes[gt_car_index][:, :7].view(-1, 7))
                    iou, gt_index = iou3d_rcnn.max(1)
                    car_iou.append(iou.view(-1, 1))
                    cor_gt = cur_batch_gt_boxes[gt_car_index][gt_index][:, :7].view(-1, 7)
                    car.append(torch.cat([pred_boxes[car_index], cor_gt], dim=1))

            bus_index = pred_labels == 2
            gt_bus_index = cur_batch_gt_boxes[:, -1] == 2
            bus_number += bus_index.sum()
            gt_bus_number += gt_bus_index.sum()
            if bus_index.sum() > 0:
                if gt_bus_index.sum()==0:
                    fake_iou = torch.zeros(bus_index.sum(), 1, device=gt_boxes.device)
                    bus_iou.append(fake_iou)
                    fake_gt = torch.zeros_like(pred_boxes[bus_index]).reshape(-1, 7)
                    bus.append(torch.cat([pred_boxes[bus_index], fake_gt], dim=1))
                else:
                    iou3d_rcnn = boxes_iou3d_gpu(pred_boxes[bus_index], cur_batch_gt_boxes[gt_bus_index][:, :7].view(-1, 7))
                    iou, gt_index = iou3d_rcnn.max(1)
                    bus_iou.append(iou.view(-1, 1))
                    cor_gt = cur_batch_gt_boxes[gt_bus_index][gt_index][:, :7].view(-1, 7)
                    bus.append(torch.cat([pred_boxes[bus_index], cor_gt], dim=1))

            truck_index = pred_labels == 3
            gt_truck_index = cur_batch_gt_boxes[:, -1] == 3
            truck_number += truck_index.sum()
            gt_truck_number += gt_truck_index.sum()
            if truck_index.sum() > 0:
                if gt_truck_index.sum()==0:
                    fake_iou = torch.zeros(truck_index.sum(), 1, device=gt_boxes.device)
                    truck_iou.append(fake_iou)
                    fake_gt = torch.zeros_like(pred_boxes[truck_index]).reshape(-1, 7)
                    truck.append(torch.cat([pred_boxes[truck_index], fake_gt], dim=1))
                else:
                    iou3d_rcnn = boxes_iou3d_gpu(pred_boxes[truck_index], cur_batch_gt_boxes[gt_truck_index][:, :7].view(-1, 7))
                    iou, gt_index = iou3d_rcnn.max(1)
                    truck_iou.append(iou.view(-1, 1))
                    cor_gt = cur_batch_gt_boxes[gt_index][:, :7].view(-1, 7)
                    truck.append(torch.cat([pred_boxes[truck_index], cor_gt], dim=1))
        if cfg.LOCAL_RANK == 0:
            progress_bar.update()
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    car_iou = torch.cat(car_iou)
    car = torch.cat(car)
    bus = torch.cat(bus)
    bus_iou = torch.cat(bus_iou)
    truck = torch.cat(truck)
    truck_iou = torch.cat(truck_iou)

    bus_iou_index = []
    bus_iou_5055_index = (bus_iou>=0.50) & (bus_iou < 0.55)
    bus_iou_5560_index = (bus_iou>=0.55) & (bus_iou < 0.60)
    bus_iou_6065_index = (bus_iou>=0.60) & (bus_iou < 0.65)
    bus_iou_6570_index = (bus_iou>=0.65) & (bus_iou < 0.70)
    bus_iou_70_index = bus_iou>=0.70
    bus_iou_index.append(bus_iou_5055_index.squeeze())
    bus_iou_index.append(bus_iou_5560_index.squeeze())
    bus_iou_index.append(bus_iou_6065_index.squeeze())
    bus_iou_index.append(bus_iou_6570_index.squeeze())
    bus_iou_index.append(bus_iou_70_index.squeeze())
    
    truck_iou_index = []
    truck_iou_5055_index = (truck_iou>=0.50) & (truck_iou < 0.55)
    truck_iou_5560_index = (truck_iou>=0.55) & (truck_iou < 0.60)
    truck_iou_6065_index = (truck_iou>=0.60) & (truck_iou < 0.65)
    truck_iou_6570_index = (truck_iou>=0.65) & (truck_iou < 0.70)
    truck_iou_70_index = truck_iou>=0.70
    truck_iou_index.append(truck_iou_5055_index.squeeze())
    truck_iou_index.append(truck_iou_5560_index.squeeze())
    truck_iou_index.append(truck_iou_6065_index.squeeze())
    truck_iou_index.append(truck_iou_6570_index.squeeze())
    truck_iou_index.append(truck_iou_70_index.squeeze())

    for i in range(len(truck_iou_index)):
        x2 = (truck[truck_iou_index[i].squeeze()][..., 0] - truck[truck_iou_index[i].squeeze()][..., 7])**2
        y2 = (truck[truck_iou_index[i].squeeze()][..., 1] - truck[truck_iou_index[i].squeeze()][..., 8])**2
        offset = torch.sqrt(x2+y2) / truck_iou_index[i].sum() 
        print('truck:', torch.sum(torch.abs(truck[truck_iou_index[i].squeeze()][..., [3, 4, 5, 6]] - truck[truck_iou_index[i].squeeze()][..., [10, 11, 12, 13]]), dim=0) / truck_iou_index[i].sum(), ' ', 
        offset)

    for i in range(len(bus_iou_index)):
        print('bus:', torch.sum(torch.abs(bus[bus_iou_index[i].squeeze()][..., [3, 4, 5, 6]] - bus[bus_iou_index[i].squeeze()][..., [10, 11, 12, 13]]), dim=0) / bus_iou_index[i].sum(), ' ', 
        (torch.sqrt((bus[bus_iou_index[i].squeeze()][..., 0] - bus[bus_iou_index[i].squeeze()][..., 7])**2 + (bus[bus_iou_index[i].squeeze()][..., 1] - bus[bus_iou_index[i].squeeze()][..., 8])**2)).sum() / bus_iou_index[i].sum())
        
    return ret_dict

if __name__ == '__main__':
    pass
