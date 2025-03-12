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


def compute_errors(flow_gt, flow_preds, valid_arr):
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
    a1_arr = []
    a2_arr = []
    a3_arr = []
    abs_rel_arr = []
    rmse_arr = []
    log_10_arr = []
    rmse_log_arr = []
    silog_arr = []
    sq_rel_arr = []
    min_depth_eval = 0.001
    max_depth_eval = 80

    

    for gt, pred, valid in zip(flow_gt,flow_preds,valid_arr):
        
        gt = gt.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        valid = valid.squeeze().cpu()
        # print(gt.max(),gt.min())
        # print(pred.max(),pred.min())
        gt, pred= gt[valid.bool()], pred[valid.bool()]


        #median scaling
        ratio = np.median(gt) / np.median(pred)
        pred *= ratio

        pred[pred < min_depth_eval] = min_depth_eval
        pred[pred > max_depth_eval] = max_depth_eval
        pred[np.isinf(pred)] = max_depth_eval
        pred[np.isnan(pred)] = min_depth_eval

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = (np.abs(gt - pred) / gt).mean()
        sq_rel =(((gt - pred) ** 2) / gt).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        
        a1_arr.append(a1)
        a2_arr.append(a2)
        a3_arr.append(a3)
        abs_rel_arr.append(abs_rel)
        rmse_arr.append(rmse)
        log_10_arr.append(log_10)
        rmse_log_arr.append(rmse_log)
        silog_arr.append(silog)
        sq_rel_arr.append(sq_rel)

    a1_arr_mean = sum(a1_arr) / len(a1_arr)
    a2_arr_mean = sum(a2_arr) / len(a2_arr)
    a3_arr_mean = sum(a3_arr) / len(a3_arr)
    abs_rel_arr_mean = sum(abs_rel_arr) / len(abs_rel_arr)
    rmse_arr_mean = sum(rmse_arr) / len(rmse_arr)
    log_10_arr_mean = sum(log_10_arr) / len(log_10_arr)
    rmse_log_arr_mean = sum(rmse_log_arr) / len(rmse_log_arr)
    silog_arr_mean = sum(silog_arr) / len(silog_arr)
    sq_rel_arr_mean = sum(sq_rel_arr) / len(sq_rel_arr)


    return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, log_10=log_10_arr_mean, rmse_log=rmse_log_arr_mean,
                silog=silog_arr_mean, sq_rel=sq_rel_arr_mean)
    #return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, sq_rel=sq_rel_arr_mean)




def calc_metric_avg(metrics_arr):
    len_arr = len(metrics_arr)
    metric_avg = {'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'abs_rel': 0.0, 'rmse': 3.0, 'log_10': 0.0, 'rmse_log': 0.0, 'silog': 0.0, 'sq_rel': 0.0}
    for metric in metrics_arr:
        for key in metric_avg.keys():
            metric_avg[key] = metric_avg[key] + metric[key]

    for key in metric_avg.keys():
        metric_avg[key] = metric_avg[key]/len_arr
    
    return metric_avg



@torch.no_grad()
def validate_kitti(model, seg_any_predictor: SamPredictor, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(seg_any_predictor,aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    metrics_arr = []
    pass_num = 0
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        # if(flow_gt.max()>80):
        #     print(f'{val_id} pass')
        #     pass_num += 1
        #     continue
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        #padder = InputPadder(image1.shape, divis_by=32)
        #image1, image2 = padder.pad(image1, image2)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)
            end = time.time()

        #flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        if flow_pr.shape[-2:] != flow_gt.shape[-2:]:
            flow_pr = nn.functional.interpolate(
                flow_pr, size=flow_gt.shape[-2:], mode="bilinear", align_corners=True
            )
        flow_gt = flow_gt.unsqueeze(0)
        valid_gt = valid_gt.unsqueeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)


        # flow_pr = flow_pr * 80
        # flow_pr = ((flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min())) * 255
        # flow_gt = ((flow_gt - flow_gt.min()) / (flow_gt.max() - flow_gt.min())) * 255
        metrics = compute_errors(flow_gt, flow_pr,valid_gt)
        metrics_arr.append(metrics)
        if val_id < 9 or (val_id+1)%10 == 0:
            print(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. {metrics}")


        # '''Inference 결과 저장 코드'''
        # outdir = './AsymKD_inference_result'
        # if metrics['a1']>=0.9:
        #     flow_pr = flow_pr.squeeze()
        #     flow_pr = (flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min()) * 255.0
        #     flow_pr = flow_pr.cpu().numpy().astype(np.uint8)
        #     flow_pr = cv2.applyColorMap(flow_pr, cv2.COLORMAP_INFERNO)
            
        #     if metrics['a1']>=0.95:
        #         cv2.imwrite(os.path.join(outdir, '###AsymKD_Feas_'+str(val_id) + '_depth.png'), flow_pr)
        #     else:
        #         cv2.imwrite(os.path.join(outdir, 'AsymKD_Feas_'+str(val_id) + '_depth.png'), flow_pr)

        # outdir = './AsymKD_inference_input'
        # if metrics['a1']>=0.9:
        #     input_image = (image1 - image1.min()) / (image1.max() - image1.min()) * 255.0
        #     input_image = input_image.cpu().numpy().astype(np.uint8)
        #     input_image = input_image[0].transpose(1, 2, 0)
        #     input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        #     if metrics['a1']>=0.95:
        #         cv2.imwrite(os.path.join(outdir, '###AsymKD_Feas_'+str(val_id) + '_input.png'), input_image)
        #     else:
        #         cv2.imwrite(os.path.join(outdir, 'AsymKD_Feas_'+str(val_id) + '_input.png'), input_image)
        
        # 낮은 성능 디버그 코드
        # if metrics['a1'] >= 0.9:
        #     plt.imshow(flow_pr.squeeze().cpu().detach().numpy(), cmap='viridis')  # viridis 컬러 맵을 사용하여 시각화
        #     plt.colorbar(label='Depth Value')  # 컬러 바 추가
        #     plt.title('Depth Image Visualization')
        #     plt.xlabel('X-axis')
        #     plt.ylabel('Y-axis')
        #     plt.savefig(f'./AsymKD_inference_result/###depth_image_visualization{val_id}.png')
        #     plt.close()

        #     plt.imshow(flow_gt.squeeze().cpu().detach().numpy(), cmap='viridis')  # viridis 컬러 맵을 사용하여 시각화
        #     plt.colorbar(label='Depth Value')  # 컬러 바 추가
        #     plt.title('Ground Truth Image Visualization')
        #     plt.xlabel('X-axis')
        #     plt.ylabel('Y-axis')
        #     plt.savefig(f'./AsymKD_inference_input/###Ground_Truth_image_visualization{val_id}.png')
        #     plt.close()


    print(f'pass_num : {pass_num}')
    return calc_metric_avg(metrics_arr)



@torch.no_grad()
def validate_kitti_for_depth_anything(model, seg_any_predictor: SamPredictor, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(seg_any_predictor,aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    metrics_arr = []
    pass_num = 0
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        
        # if(flow_gt.max()>80):
        #     print(f'{val_id} pass')
        #     pass_num += 1
        #     continue
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        #padder = InputPadder(image1.shape, divis_by=32)
        #image1, image2 = padder.pad(image1, image2)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1)
            end = time.time()

        #flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        if flow_pr.shape[-2:] != flow_gt.shape[-2:]:
            flow_pr = nn.functional.interpolate(
                flow_pr, size=flow_gt.shape[-2:], mode="bilinear", align_corners=True
            )
        flow_gt = flow_gt.unsqueeze(0)
        valid_gt = valid_gt.unsqueeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        #print(flow_pr.min(),flow_pr.max())
        # flow_pr = ((flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min() + 1e-6)) * 80
        # flow_gt = ((flow_gt - flow_gt.min()) / (flow_gt.max() - flow_gt.min() + 1e-6))
        metrics = compute_errors(flow_gt, flow_pr,valid_gt)
        metrics_arr.append(metrics)
        if val_id < 9 or (val_id+1)%10 == 0:
            print(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. {metrics}")
        # if val_id < 9 or (val_id+1)%10 == 0:
        #     logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. {metrics}")

        # '''Inference 결과 저장 코드'''
        # outdir = './Depth_Anything_inference_result'
        # flow_pr = flow_pr.squeeze()
        # flow_pr = (flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min()) * 255.0
        # flow_pr = flow_pr.cpu().numpy().astype(np.uint8)
        # flow_pr = cv2.applyColorMap(flow_pr, cv2.COLORMAP_INFERNO)
        # cv2.imwrite(os.path.join(outdir, 'Depth_Anything_'+str(val_id) + '_depth.png'), flow_pr)

        # '''Inference ground truth 저장 코드'''
        # flow_gt = flow_gt.squeeze()
        # flow_gt = (flow_gt - flow_gt.min()) / (flow_gt.max() - flow_gt.min()) * 255.0
        # flow_gt = flow_gt.cpu().numpy().astype(np.uint8)
        # flow_gt = cv2.applyColorMap(flow_gt, cv2.COLORMAP_INFERNO)
        # cv2.imwrite(os.path.join(outdir, 'Depth_Anything_'+str(val_id) + '_depth_gt.png'), flow_gt)
        
    return calc_metric_avg(metrics_arr)


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
                print(new_state_dict)
            model.to(rank)
            model.eval()
            AsymKD_metric = validate_kitti(model,segment_anything_predictor)

            Depth_Any_metric = {'a1': 0.8610402657398244, 'a2': 0.9572192992174944, 'a3': 0.9794083171064192, 'abs_rel': 0.1272420384734869, 'rmse': 5.422292325198651, 'log_10': 0.0554339356161654, 'rmse_log': 0.21866457603871822, 'silog': 21.417029822075456, 'sq_rel': 1.0769053649902345}


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
            filename = 'eval_result.txt'
            with open(filename, 'a') as a:
                # 새파일에 이어서 쓰기
                a.write(f'{print_str}\n')

    finally:
        cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    manager = Manager()
    queue = manager.Queue()    
    start_num = 1
    end_num = 187
        
    for i in range(end_num,start_num-1,-1):
        queue.put(f'checkpoints_new_loss/{i}000_AsymKD_new_loss.pth')
    mp.spawn(eval, args=(world_size,queue,), nprocs=world_size, join=True)