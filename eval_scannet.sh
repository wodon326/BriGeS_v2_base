set -e
set -x

# least_square : Marigold
# least_square_disparity : Depth Anything, BFM, MiDaS, DPT
# BFM needs a checkpoint -> --bfm_checkpoint

CUDA_VISIBLE_DEVICES=4 python AsymKD_evaluate_affine_inv_gpu.py \
    --model dpt_large \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir output/scannet
