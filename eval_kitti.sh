set -e
set -x

# least_square : Marigold
# least_square_disparity : Depth Anything, BFM, MiDaS, DPT
# BFM needs a checkpoint -> --bfm_checkpoint

CUDA_VISIBLE_DEVICES=7 python AsymKD_evaluate_affine_inv_gpu.py \
    --model marigold \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --output_dir output/kitti_eigen_test
