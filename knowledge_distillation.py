from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from AsymKD.dpt import AsymKD_DepthAnything

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from AsymKD_student import AsymKD_Student
from AsymKD_evaluate import *
import core.AsymKD_datasets as datasets
from segment_anything import sam_model_registry, SamPredictor
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
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


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

        gt = gt.squeeze().cpu().numpy()
        pred = pred.clone().squeeze().cpu().detach().numpy()
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

def sequence_loss(flow_preds, flow_gt, valid, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # L1 loss
    flow_loss = F.l1_loss(flow_preds[valid.bool()], flow_gt[valid.bool()])

    # for i in range(n_predictions):
    #     assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
    #     # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
    #     flow_preds[i] = flow_preds[i].squeeze(0)
    #     i_loss = (flow_preds[i] - flow_gt[i]).abs()
    #     assert i_loss.shape == valid[i].shape, [i_loss.shape, valid[i].shape, flow_gt.shape, flow_preds[i].shape]
    #     flow_loss += i_loss[valid[i].bool()].mean()

    # epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    # epe = epe.view(-1)[valid.view(-1)]

    # metrics = {
    #     'epe': epe.mean().item(),
    #     '1px': (epe < 1).float().mean().item(),
    #     '3px': (epe < 3).float().mean().item(),
    #     '5px': (epe < 5).float().mean().item(),
    # }

    metrics = compute_errors(flow_gt, flow_preds, valid)

    # print("############")
    # print(metrics)
    
    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #         pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, 
            div_factor=1, final_div_factor=10000, 
            pct_start=0.7, three_phase=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def train(rank, world_size, args):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = "sam_vit_l_0b3195.pth"
        model_type = "vit_l"
        segment_anything = sam_model_registry[model_type](checkpoint=checkpoint).to(rank).eval()
        segment_anything_predictor = SamPredictor(segment_anything)

        for child in segment_anything.children():
            ImageEncoderViT = child
            break
        asymkd_moe = AsymKD_DepthAnything(ImageEncoderViT = ImageEncoderViT).to(rank)
        teacher_model = DDP(asymkd_moe, device_ids=[rank], find_unused_parameters=True)
        teacher_model.eval()


        
        AsymKD = AsymKD_Student().to(rank)
        student_model = DDP(AsymKD, device_ids=[rank], find_unused_parameters=True)
        student_model.train()
        
        M = 1000000
        print("Parameter Count: %dM" % (int(count_parameters(student_model))/M))
        print("Solution 3 KD")
        train_loader = datasets.fetch_dataloader(args,segment_anything_predictor, rank, world_size)
        optimizer, scheduler = fetch_optimizer(args, student_model)
        total_steps = 0
        logger = Logger(student_model, scheduler)
        restore_ckpt = 'checkpoints/train_moe_AsymKD.pth'
        if restore_ckpt is not None:
            assert restore_ckpt.endswith(".pth")
            logging.info("Loading checkpoint...")
            checkpoint = torch.load(restore_ckpt, map_location=torch.device('cuda', rank))
            teacher_model.load_state_dict(checkpoint, strict=True)
            logging.info(f"Done loading checkpoint")

        #model.module.freeze_bn() # We keep BatchNorm frozen

        validation_frequency = 10000
        scaler = GradScaler(enabled=args.mixed_precision)




        stored_teacher_features = []
        stored_depth_images = []

        global_batch_num = 0
        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            
            optimizer.zero_grad()
            depth_image, seg_image, flow, valid = [x.cuda() for x in data_blob]

            with torch.no_grad():
                teacher_feature, _ = teacher_model(depth_image, seg_image)
                stored_teacher_features.append([teach_feat.cpu() for teach_feat in teacher_feature])
                stored_depth_images.append(depth_image.cpu())
            

            depth_image_h, depth_image_w = depth_image.shape[-2:]
            depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14

            student_feature = student_model(depth_image,depth_patch_h, depth_patch_w)
            loss = None
            for teacher, student in zip(teacher_feature, student_feature):
                if(loss is not None):
                    loss += F.mse_loss(student, teacher)
                else:
                    loss = F.mse_loss(student, teacher)
            
            if rank == 0:
                logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)

            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if(i_batch%1000==999):
                for _ in range(9):
                    for img ,feat in zip(stored_depth_images,stored_teacher_features):
                        teach_feat = [f.to(rank) for f in feat]
                        img = img.to(rank)
                        depth_image_h, depth_image_w = img.shape[-2:]
                        depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14

                        student_feature = student_model(img, depth_patch_h, depth_patch_w)

                        loss = None
                        for teacher, student in zip(teach_feat, student_feature):
                            if(loss is not None):
                                loss += F.mse_loss(student, teacher)
                            else:
                                loss = F.mse_loss(student, teacher)
                        
                        if rank == 0:
                            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)

                        global_batch_num += 1
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

                        scaler.step(optimizer)
                        scheduler.step()
                        scaler.update()
                del stored_teacher_features
                del stored_depth_images
                stored_depth_images = []
                stored_teacher_features= []
                torch.cuda.empty_cache()
                gc.collect()
                if rank == 0:
                    save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save(student_model.state_dict(), save_path)

            if total_steps%100==0:
                torch.cuda.empty_cache()
                gc.collect()
            total_steps += 1

        for _ in range(9):
            for img ,feat in zip(stored_depth_images,stored_teacher_features):
                teach_feat = [f.to(rank) for f in feat]
                img = img.to(rank)
                depth_image_h, depth_image_w = img.shape[-2:]
                depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14
                student_feature = student_model(img,depth_patch_h, depth_patch_w)

                loss = None
                for teacher, student in zip(teach_feat, student_feature):
                    if(loss is not None):
                        loss += F.mse_loss(student, teacher)
                    else:
                        loss = F.mse_loss(student, teacher)
                
                if rank == 0:
                    logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)

                global_batch_num += 1
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
        del stored_teacher_features
        del stored_depth_images
        stored_depth_images = []
        stored_teacher_features= []
        torch.cuda.empty_cache()
        gc.collect()
        
        if rank == 0:
            save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path.absolute()}")
            torch.save(student_model.state_dict(), save_path)
            

        print("FINISHED TRAINING")
        logger.close()
        PATH = 'checkpoints/%s.pth' % args.name
        torch.save(student_model.state_dict(), PATH)

        return PATH
    finally:
        cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='AsymKD_MOEKD', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['tartan_air'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.00005, help="max learning rate.")
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

    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,args,), nprocs=world_size, join=True)
