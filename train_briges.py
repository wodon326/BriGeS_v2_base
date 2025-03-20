from __future__ import print_function, division
import os
import argparse
import logging
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from AsymKD.depth_latent4_kd_naive_split_adapter import kd_naive_depth_latent4_split_adapter
from AsymKD.depth_latent4_kd_naive_split_adapter_cbam import kd_naive_depth_latent4_split_adapter_cbam
from AsymKD.BriGeS_cbam import BriGeS_cbam
from AsymKD.BriGeS_cbam_high_only import BriGeS_cbam_high_only
from segment_anything import  sam_model_registry, SamPredictor

from core.loss import GradL1Loss, ScaleAndShiftInvariantLoss
from dataset.util.alignment_gpu import align_depth_least_square

import core.AsymKD_datasets as datasets
import gc

import torch.nn.functional as F
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
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
    # log_10_arr = []
    # rmse_log_arr = []
    # silog_arr = []
    sq_rel_arr = []

    min_depth_eval = 0.0001
    max_depth_eval = 1

    for gt, pred, valid in zip(flow_gt, flow_preds, valid_arr):
        
        disparity_pred, scale, shift = align_depth_least_square(
            gt_arr=gt,
            pred_arr=pred,
            valid_mask_arr=valid,
            return_scale_shift=True,
        )
        gt = gt.squeeze().cpu().numpy()
        pred = disparity_pred.clone().squeeze().cpu().detach().numpy()
        valid = valid.squeeze().cpu()             
        pred[pred < min_depth_eval] = min_depth_eval
        pred[pred > max_depth_eval] = max_depth_eval
        pred[np.isinf(pred)] = max_depth_eval
        pred[np.isnan(pred)] = min_depth_eval

        # pred[pred < min_depth_eval] = min_depth_eval
        # pred[pred > max_depth_eval] = max_depth_eval
        # pred[np.isinf(pred)] = max_depth_eval
        # pred[np.isnan(pred)] = min_depth_eval
        # gt[gt < min_depth_eval] = min_depth_eval
        # gt[gt > max_depth_eval] = max_depth_eval
        # gt[np.isinf(gt)] = max_depth_eval
        # gt[np.isnan(gt)] = min_depth_eval

        gt, pred= gt[valid.bool()], pred[valid.bool()]

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = (np.abs(gt - pred) / gt).mean()
        sq_rel =(((gt - pred) ** 2) / gt).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        # rmse_log = (np.log(gt) - np.log(pred)) ** 2
        # rmse_log = np.sqrt(rmse_log.mean())

        # err = np.log(pred) - np.log(gt)
        # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        # log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        a1_arr.append(a1)
        a2_arr.append(a2)
        a3_arr.append(a3)
        abs_rel_arr.append(abs_rel)
        rmse_arr.append(rmse)
        # log_10_arr.append(log_10)
        # rmse_log_arr.append(rmse_log)
        # silog_arr.append(silog)
        sq_rel_arr.append(sq_rel)

    a1_arr_mean = sum(a1_arr) / len(a1_arr)
    a2_arr_mean = sum(a2_arr) / len(a2_arr)
    a3_arr_mean = sum(a3_arr) / len(a3_arr)
    abs_rel_arr_mean = sum(abs_rel_arr) / len(abs_rel_arr)
    rmse_arr_mean = sum(rmse_arr) / len(rmse_arr)
    # log_10_arr_mean = sum(log_10_arr) / len(log_10_arr)
    # rmse_log_arr_mean = sum(rmse_log_arr) / len(rmse_log_arr)
    # silog_arr_mean = sum(silog_arr) / len(silog_arr)
    sq_rel_arr_mean = sum(sq_rel_arr) / len(sq_rel_arr)

    # return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, log_10=log_10_arr_mean, rmse_log=rmse_log_arr_mean,
    #             silog=silog_arr_mean, sq_rel=sq_rel_arr_mean)
    return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, sq_rel=sq_rel_arr_mean)

def sequence_loss(flow_preds, flow_gt, valid):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid]).any()
    # print(valid.dtype, valid.shape)
    # print(flow_preds.shape, flow_gt.shape)
    # L1 loss
    flow_loss = F.l1_loss(flow_preds[valid], flow_gt[valid])
    metrics = compute_errors(flow_gt, flow_preds, valid)

    
    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #         pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if(args.train_style == 'cnn'):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, 
            div_factor=1, final_div_factor=10000, 
            pct_start=0.7, three_phase=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    def __init__(self, rank, save_dir, keys):
        self.metric_history = {k: {} for k in keys}
        self.writer = SummaryWriter(log_dir=f'runs/{save_dir}') if rank == 0 else None

    def push(
        self, 
        metrics: dict[str, float], 
        model_type: str,
    ):
        if self.writer is None: return
        assert model_type in self.metric_history.keys(), f"Invalid model type: {model_type}"
        for key in metrics:
            if key not in self.metric_history[model_type]:
                self.metric_history[model_type][key] = 0.0
            self.metric_history[model_type][key] += metrics[key]
    
    def flush(self, model_type: str):
        if self.writer is None: return
        assert model_type in self.metric_history.keys(), f"Invalid model type: {model_type}"
        self.metric_history[model_type] = {}
    
    def upload(self, model_type: str, total_samples: int, total_steps: int):
        if self.writer is None: return
        for k in self.metric_history[model_type]:
            self.add_scalar(f"metrics_kd_more_data/{k}/{model_type}", self.metric_history[model_type][k]/total_samples, total_steps)

    def close(self):
        if self.writer is None: return
        self.writer.close()
    
    def add_scalar(self, name, value, step):
        if self.writer is None: return
        self.writer.add_scalar(name, value, step)
    
    def add_image(self, name, image, step):
        if self.writer is None: return
        self.writer.add_image(name, image, step)

class State(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def capture(self):
        return {
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
    
    def apply_snapshot(self, obj):
        self.model.module.load_state_dict(obj['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(obj['optimizer_state_dict'])
        # self.scheduler.load_state_dict(obj['scheduler_state_dict'])
    
    def save(self, path):
        torch.save(self.capture(), path)
    
    def load(self, path, device):
        obj = torch.load(path, map_location=device)
        self.apply_snapshot(obj)

def train(rank, world_size, args):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        # init scalars
        save_step = 500
        total_steps = 0
        epoch = 0
        # loss parameter
        # alpha = 0.2

        checkpoint = "BriGeS_checkpoints/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        segment_anything = sam_model_registry[model_type](checkpoint=checkpoint).to(rank).eval()
        segment_anything_predictor = SamPredictor(segment_anything.to('cpu'))
        for child in segment_anything.children():
            ImageEncoderViT = child
            break
        BriGeS = BriGeS_cbam_high_only(ImageEncoderViT = ImageEncoderViT).to(rank)

        checkpoint = "BriGeS_checkpoints/depth_anything_vitb14.pth"
        new_state_dict = BriGeS.load_ckpt(checkpoint, device=torch.device('cuda', rank))
        BriGeS.freeze_BriGeS_cbam_style()
        if rank == 0:
            logging.info(f"loading backbones from {checkpoint}")
            print('BriGeS : ', new_state_dict.keys())

        if rank == 0:
            for n, p in BriGeS.named_parameters():
                print(f'{n} : {p.requires_grad}')
        BriGeS = torch.nn.SyncBatchNorm.convert_sync_batchnorm(BriGeS)
        #model.module.freeze_bn() # We keep BatchNorm frozen
        model = DDP(BriGeS, device_ids=[rank], find_unused_parameters=False)
        model.train()
        
        if rank == 0:
            print(f"Parameter Count: {count_parameters(model)}")
            print(f"Trainable Parameter Count: {count_trainable_parameters(model)}")
            logging.info(f"Parameter Count: {count_parameters(model)}")
            logging.info("AsymKD_VIT Train")
        
        # load others
        train_loader, val_loader = datasets.fetch_dataloader(args,segment_anything_predictor, rank, world_size)
        optimizer, scheduler = fetch_optimizer(args, model)
        scaler = GradScaler(enabled=args.mixed_precision)
        logger = Logger(rank, args.save_dir, keys=["train_BriGeS", "val_BriGeS"])

        state = State(model, optimizer, scheduler)

        # load loss
        SSILoss = ScaleAndShiftInvariantLoss()
        grad_loss = GradL1Loss()

        # load snapshot
        if args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth")
            state.load(args.restore_ckpt, torch.device('cuda', rank))
            total_steps = 90500

        while total_steps < args.num_steps:
            for i_batch, data_blob in enumerate(pbar := tqdm(train_loader)):
                pbar.set_description(f"Epoch {epoch}")
                # if(pass_num>0):
                #     pass_num -= 1
                #     continue
                optimizer.zero_grad()
                depth_image, seg_image , flow, valid = [x.cuda() for x in data_blob]
                assert model.training
                flow_predictions = model(depth_image, seg_image)
                assert model.training
                # loss, metrics = sequence_loss(flow_predictions, flow, valid)

                try:
                    l_si, scaled_pred = SSILoss(
                        flow_predictions, flow, mask=valid.bool(), interpolate=True, return_interpolated=True)
                    loss = l_si
                    # l_grad = grad_loss(scaled_pred, flow, mask=valid.bool().unsqueeze(1))
                    # loss = loss + 0.001 * l_grad
                except Exception as e:
                    # loss, _ = sequence_loss(flow_predictions, flow, valid)
                    loss = torch.tensor(0.0).cuda()
                    filename = 'Exception_catch.txt'
                    a = open(filename, 'a')
                    a.write(str(e)+'\n')
                    a.close()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                if rank == 0:
                    logger.add_scalar("live_loss", l_si.item(), total_steps)
                    logger.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], total_steps)
                    _ , metrics = sequence_loss(flow_predictions, flow, valid.bool().unsqueeze(1))
                    logger.push(metrics, "train_BriGeS")

                    if(total_steps % 10 == 10-1):
                        # inference visualization in tensorboard while training
                        rgb = depth_image[0].cpu().detach().numpy()
                        rgb = ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))) * 255
            
                        gt = flow[0].cpu().detach().numpy()
                        gt = ((gt - np.min(gt)) / (np.max(gt) - np.min(gt))) * 255
            
                        pred = flow_predictions[0].cpu().detach().numpy()
                        pred = ((pred - np.min(pred)) / (np.max(pred) - np.min(pred))) * 255
                        
                        logger.add_image('RGB', rgb.astype(np.uint8), total_steps)
                        logger.add_image('GT', gt.astype(np.uint8), total_steps)
                        logger.add_image('Prediction/student', pred.astype(np.uint8), total_steps)

                    if total_steps % save_step == save_step-1:
                        save_path = Path(f"checkpoint_{args.save_dir}/{total_steps + 1}_{args.name}.pth")
                        logging.info(f"Saving file {save_path.absolute()}")
                        state.save(save_path)

                if total_steps%100==0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if rank == 0 and total_steps  %  5000 == 5000 - 1:
                    model.eval()
                    vbar = tqdm(val_loader)
                    vbar.set_description(f"Validation")
                    for data_blob in vbar:
                        depth_image, seg_image, flow, valid = [x.cuda() for x in data_blob]
                        h, w = depth_image.shape[-2:]
                        with torch.no_grad():
                            pred_depth = model.module(depth_image, seg_image)
                            model_type = "val_BriGeS"
                            loss, metrics = sequence_loss(pred_depth, flow, valid.bool().unsqueeze(1))
                            logger.push(metrics, model_type)

                    for model_type in ["train_BriGeS", "val_BriGeS"]:
                        logger.upload(model_type, len(val_loader.dataset), total_steps)
                        logger.flush(model_type)
                    model.train()

                total_steps += 1
            epoch += 1      

        print("FINISHED TRAINING")
        logger.close()
        state.save(f"checkpoint_{args.save_dir}/{args.name}.pth")
        return None
    finally:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='AsymKD_new_loss', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--epoch', type=int, default=3, help="length of training schedule.")
    parser.add_argument('--ckpt', type=str, help="load_ckpt")
    parser.add_argument('--student_ckpt', type=str, help="load_ckpt")
    parser.add_argument('--save_dir', type=str, help="save_dir")
    parser.add_argument('--train_style', type=str, help="train_style")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['tartan_air'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.00001, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[518, 518], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['hf','h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    Path(f"checkpoint_{args.save_dir}").mkdir(exist_ok=True, parents=True)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,args,), nprocs=world_size, join=True)
