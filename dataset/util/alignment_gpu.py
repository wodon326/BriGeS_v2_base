import numpy as np
import torch


def align_depth_least_square(
    gt_arr: torch.tensor,
    pred_arr: torch.tensor,
    valid_mask_arr: torch.tensor,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest").cuda()
            gt = downscaler(gt.unsqueeze(0)).squeeze(0)
            pred = downscaler(pred.unsqueeze(0)).squeeze(0)
            valid_mask = downscaler(valid_mask.unsqueeze(0).float()).bool().squeeze(0)

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # Torch solver using GPU
    _ones = torch.ones_like(pred_masked).cuda()
    A = torch.cat([pred_masked, _ones], axis=-1)

    X = torch.linalg.lstsq(A, gt_masked).solution
    scale, shift = X.squeeze()

    aligned_pred = pred_arr * scale.item() + shift.item()

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale.item(), shift.item()
    else:
        return aligned_pred


# ******************** disparity space ********************
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth).cuda()
    elif isinstance(depth, np.ndarray):
        depth = torch.as_tensor(depth).cuda()
        disparity = torch.zeros_like(depth).cuda()
        
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
