# Last modified: 2024-03-11
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
)
from dataset.util import metric
from dataset.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from dataset.util.metric import MetricTracker

import torch.nn.functional as F
from depth_anything_for_evaluate.dpt import DepthAnything
from segment_anything import sam_model_registry, SamPredictor
from AsymKD.dpt import AsymKD_DepthAnything_Infer

@torch.no_grad()
def infer(model, image, seg_image=None, **kwargs):
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

    if seg_image is None:
        pred1 = model(image, **kwargs)
        pred2 = model(torch.flip(image, [3]), **kwargs)
    else:
        pred1 = model(image, seg_image, **kwargs)
        pred2 = model(torch.flip(image, [3]), torch.flip(seg_image, [3]), **kwargs)

    pred1 = get_depth_from_prediction(pred1)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred

eval_metrics = [
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "i_rmse",
    "silog_rmse",
]

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # model
    parser.add_argument(
        "--model", type=str, required=True, help="Model to evaluate."
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    # LS depth alignment
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()

    output_dir = args.output_dir

    model_type = args.model

    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir

    alignment = args.alignment
    alignment_max_res = args.alignment_max_res

    no_cuda = args.no_cuda
    pred_suffix = ".npy"

    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if "depth_anything" in model_type:
        if "small" in model_type:
            encoder = "vits"
        elif "base" in model_type:
            encoder = "vitb"
        elif "large" in model_type:
            encoder = "vitl"
        model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(device)
    elif model_type == "bfm":
        segment_anything = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").to(device)
        segment_anything_predictor = SamPredictor(segment_anything)

        for child in segment_anything.children():
            ImageEncoderViT = child
            break
        model = AsymKD_DepthAnything_Infer(ImageEncoderViT=ImageEncoderViT).to(device)
        restore_ckpt = "checkpoints_new_loss_001_smooth/61000_AsymKD_new_loss.pth"

        if restore_ckpt is not None:
            logging.info("Loading checkpoint...")
            checkpoint = torch.load(restore_ckpt, map_location=device)
            model_state_dict = model.state_dict()
            new_state_dict = {}
            for k, v in checkpoint.items():
                new_key = k.replace('module.', '')
                if new_key in model_state_dict:
                    new_state_dict[new_key] = v
    
            model_state_dict.update(new_state_dict)
            model.load_state_dict(model_state_dict)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Per-sample metric file head --------------------
    # per_sample_filename = os.path.join(output_dir, "per_sample_metrics.csv")
    # # write title
    # with open(per_sample_filename, "w+") as f:
    #     f.write("filename,")
    #     f.write(",".join([m.__name__ for m in metric_funcs]))
    #     f.write("\n")

    # -------------------- Evaluate --------------------
    model.eval()
    for data in tqdm(dataloader, desc="Evaluating"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_float = data["rgb_float"].to(device)
        rgb = data["rgb_norm"].to(device)

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        depth_raw_ts = depth_raw_ts.to(device)
        valid_mask_ts = valid_mask_ts.to(device)

        # Get prediction
        # (352, 1216)
        if "kitti" in dataset_config:
            pred_size = (518, 1792)
        # (480, 640), (768, 1024), (480, 640)
        elif "nyu" in dataset_config or "diode" in dataset_config or "scannet" in dataset_config:
            pred_size = (518, 686)
        # (4032, 6048)
        elif "eth3d" in dataset_config:
            pred_size = (518, 770)

        rgb_resized = F.interpolate(rgb, size=pred_size, mode='bilinear', align_corners=False)
        rgb_float_resized = F.interpolate(rgb_float, size=pred_size, mode='bilinear', align_corners=False)
        if "depth_anything" in model_type:
            pred = infer(model, rgb_resized)
        elif model_type == "bfm":
            rgb_resized_seg = segment_anything_predictor.set_image(rgb_float_resized.squeeze())
            pred = infer(model, rgb_resized, rgb_resized_seg)
            # with torch.no_grad():
            #     pred = model(rgb_resized, rgb_resized_seg)

        depth_pred_ts = F.interpolate(pred, size=depth_raw_ts.shape, mode='bilinear', align_corners=False)
        depth_pred = depth_pred_ts.squeeze().detach().cpu().numpy()

        # Align with GT using least square
        if "least_square" == alignment:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
        elif "least_square_disparity" == alignment:
            # convert GT depth -> GT disparity
            gt_disparity, gt_non_neg_mask = depth2disparity(
                depth=depth_raw, return_mask=True
            )
            # LS alignment in disparity space
            pred_non_neg_mask = depth_pred > 0
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

            disparity_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_disparity,
                pred_arr=depth_pred,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
            # convert to depth
            disparity_pred = np.clip(
                disparity_pred, a_min=1e-3, a_max=None
            )  # avoid 0 disparity
            depth_pred = disparity2depth(disparity_pred)

        # Clip to dataset min max
        depth_pred = np.clip(
            depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        # Evaluate (using CUDA if available)
        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += f"min_depth = {dataset.min_depth}\n"
    eval_text += f"max_depth = {dataset.max_depth}\n"

    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = f"eval_metrics-{model_type}.txt"

    _save_to = os.path.join(output_dir, metrics_filename)
    print(eval_text)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")