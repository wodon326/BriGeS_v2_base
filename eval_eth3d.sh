set -e
set -x

# least_square : Marigold
# least_square_disparity : Depth Anything, BFM, MiDaS, DPT
# BFM needs a checkpoint -> --bfm_checkpoint

CUDA_VISIBLE_DEVICES=7 python AsymKD_evaluate_affine_inv_gpu.py \
    --model dpt_large \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir output/eth3d \
    --alignment_max_res 1024