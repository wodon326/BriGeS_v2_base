from __future__ import print_function, division
import sys
sys.path.append('core')
import os
import cv2

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
#from raft_stereo import RAFTStereo, autocast
import dataset_raw_kitti
from torch.utils.data import DataLoader

import core.AsymKD_datasets as datasets
from core.utils import InputPadder
from segment_anything import  sam_model_registry, SamPredictor
from AsymKD.dpt import AsymKD_DepthAnything, AsymKD_DepthAnything_Infer
from torch.multiprocessing import Manager
import torch.nn as nn
from depth_anything_for_evaluate.dpt import DepthAnything
import torch.distributed as dist
import torch.multiprocessing as mp

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


 
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0
 
    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1
 
    def get_value(self):
        return self.avg
 
 
class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None
 
    def update(self, new_dict):
        if new_dict is None:
            return
 
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()
 
        for key, value in new_dict.items():
            self._dict[key].append(value)
 
    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}
 
 
def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'
 
    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values
 
        gt.shape should be equal to pred.shape
 
    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
 
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
 
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
 
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
 
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
 
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    # print(f'a1 : {a1}')
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)
 
 
def compute_metrics(gt, pred, interpolate=True, garg_crop=True, eigen_crop=False, dataset='kitti', min_depth_eval=1e-3, max_depth_eval=80):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)
    
    pred = pred.squeeze().cpu().numpy()
    # pred[pred < min_depth_eval] = min_depth_eval
    # pred[pred > max_depth_eval] = max_depth_eval
    # pred[np.isinf(pred)] = max_depth_eval
    # pred[np.isnan(pred)] = min_depth_eval
 
    gt_depth = gt.squeeze().cpu().numpy()

    pred = 1 / pred

    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)
 
    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)
 
        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
 
        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
 
    # median scaling
    pred = pred[valid_mask]
    gt_depth = gt_depth[valid_mask]
 
    ratio = np.median(gt_depth) / np.median(pred)
    pred *= ratio
 
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval
 
    # return compute_errors(gt_depth[valid_mask], pred[valid_mask])
    return compute_errors(gt_depth, pred)
 
 
@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred
 
    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)
 
    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])
 
    mean_pred = 0.5 * (pred1 + pred2)
 
    return mean_pred
 
 
@torch.no_grad()
def validate_kitti(model, seg_any_predictor: SamPredictor, round_vals=True, round_precision=3):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(seg_any_predictor=seg_any_predictor, aug_params={}, image_set='training')
    torch.backends.cudnn.benchmark = True
    metrics = RunningAverageDict()
 
    for sample in tqdm(val_dataset, total=len(val_dataset)):
        img_depth, img_seg, gt, valid = sample
        img_depth = img_depth[None].cuda()
        img_seg = img_seg[None].cuda()
 
        # pred = infer()
 
        flow_pr = model(img_depth, img_seg) # infer?
 
        metrics.update(compute_metrics(gt, flow_pr))
 
    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
   
    return metrics

@torch.no_grad()
def validate_kitti_for_depth_anything(model, seg_any_predictor: SamPredictor, round_vals=True, round_precision=3):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(seg_any_predictor=seg_any_predictor, aug_params={}, image_set='training')
    torch.backends.cudnn.benchmark = True
    metrics = RunningAverageDict()
 
    for val_id in tqdm(range(len(val_dataset))):
        img_depth, img_seg, gt, valid = val_dataset[val_id]
        img_depth = img_depth[None].cuda()
        img_seg = img_seg[None].cuda()
 
        # pred = infer()
 
        flow_pr = model(img_depth) # infer?
 
        metrics.update(compute_metrics(gt, flow_pr))
 
    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
   
    return metrics
 
@torch.no_grad()
def validate_raw_kitti_for_depth_anything(model, round_vals=False, round_precision=3):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    filenames = readlines(os.path.join(splits_dir, 'eigen', "test_files.txt"))
    h,w = 350, 1218
    dataset = dataset_raw_kitti.KITTIRAWDataset('../data2/kitti/', filenames,
                                           h, w,
                                           [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2,
                                pin_memory=True, drop_last=False)
    metrics = RunningAverageDict()
 
    for data in tqdm(dataloader):
        input_color = data[("color", 0, 0)].cuda()
 
        # pred = infer()
        pred_disp = model(input_color)
        # pred_disp = infer(model, input_color)
        gt = data["depth_gt"].cuda()

        # print(f'{gt.shape, pred_disp.shape}')
        metrics.update(compute_metrics(gt, pred_disp))
 
    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
   
    return metrics
 
@torch.no_grad()
def validate_raw_kitti(model, seg_any_predictor: SamPredictor, round_vals=False, round_precision=3):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    filenames = readlines(os.path.join(splits_dir, 'eigen', "test_files.txt"))
    h,w = 350, 1218
    # ../../datasets/kitti_data/
    dataset = dataset_raw_kitti.KITTIRAWDataset('../data2/kitti/', filenames,
                                           h, w,
                                           [0], 4,seg_any_predictor = seg_any_predictor, is_train=False)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2,
                                drop_last=False)
    metrics = RunningAverageDict()
 
    for data in dataloader:
        input_color = data[("color", 0, 0)].cuda()
        input_color_seg = data[("color_seg", 0, 0)].cuda()
 
        # pred = infer()
        pred_disp = model(input_color,input_color_seg)
        # pred_disp = infer(model, input_color)
        gt = data["depth_gt"].cuda()

        # print(f'{gt.shape, pred_disp.shape}')
        metrics.update(compute_metrics(gt, pred_disp))
 
    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
   
    return metrics


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

import queue


def eval(rank, world_size, queue):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        checkpoint = "sam_vit_b_01ec64.pth"
        # checkpoint = "sam_vit_l_0b3195.pth"
        model_type = "vit_b"
        segment_anything = sam_model_registry[model_type](checkpoint=checkpoint).to(rank).eval()
        segment_anything_predictor = SamPredictor(segment_anything)

        '''Depth Anything model load'''
        # encoder = 'vitb' # can also be 'vitb' or 'vitl'
        # model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(rank).eval()
        # results = validate_kitti_for_depth_anything(model,segment_anything_predictor)
        # print(f'Depth Anything evaluate result : {results}')    
        # print(f'#######Depth Anything evaluate result#############')    
        # for key in results.keys():
        #     print(f'{key} : {round(results[key], 3)}')

        #각 rank에서 5개의 모델 evaluation
        while not queue.empty():
            restore_ckpt = str(queue.get())
            print(restore_ckpt)
            torch.cuda.empty_cache()
            '''AsymKD model load'''
            for child in segment_anything.children():
                    ImageEncoderViT = child
                    break
            model = AsymKD_DepthAnything_Infer(ImageEncoderViT = ImageEncoderViT).to(rank)
            if restore_ckpt is not None:
                assert restore_ckpt.endswith(".pth")
                logging.info("Loading checkpoint...")
                checkpoint = torch.load(restore_ckpt, map_location=torch.device('cuda', rank))
                model__state_dict = model.state_dict()
                new_state_dict = {}
                for k, v in checkpoint.items():
                    # 키 매핑 규칙을 정의
                    new_key = k.replace('module.', '')  # 'module.'를 제거
                    if new_key in model__state_dict:
                        new_state_dict[new_key] = v

                model__state_dict.update(new_state_dict)
                model.load_state_dict(model__state_dict)
            if(rank == 0):
                print(new_state_dict.keys())
            model.to(rank)
            model.eval()
            AsymKD_metric = validate_raw_kitti(model,segment_anything_predictor)

            Depth_Any_metric = {'a1': 0.8796647734923237, 'a2': 0.9656859071253321, 'a3': 0.9860573899123259, 'abs_rel': 0.11549764806391348, 'rmse': 4.7324441392589325, 'log_10': 0.049063832267926426, 'rmse_log': 0.1806166836860961, 'silog': 17.612650733573325, 'sq_rel': 0.8737658289350687}


            print_str = f'#######AsymKD {restore_ckpt} evaluate result#############\n'

            for key in AsymKD_metric.keys():
                print_str += f'{key} : {round(AsymKD_metric[key], 3)}\n'

            print_str += f'#######AsymKD {restore_ckpt} diff evaluate result#############\n'
            for key in AsymKD_metric.keys():
                if(Depth_Any_metric['a1']-AsymKD_metric['a1']<0):
                    print_str += f'@@@@diff {key} : {round(Depth_Any_metric[key]-AsymKD_metric[key], 3)}\n'
                else:
                    print_str += f'diff {key} : {round(Depth_Any_metric[key]-AsymKD_metric[key], 3)}\n'
            
            print(print_str)
            filename = 'eval_result_new_0001_smooth.txt'
            with open(filename, 'a') as a:
                # 새파일에 이어서 쓰기
                a.write(f'{print_str}\n')

    finally:
        cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    manager = Manager()
    queue = manager.Queue()    
    start_num = 5
    end_num = 1870
        
    for i in range(end_num,start_num-1,-5):
        queue.put(f'checkpoints_new_loss_0001_smooth/{i}00_AsymKD_new_loss.pth')
    mp.spawn(eval, args=(world_size,queue,), nprocs=world_size, join=True)