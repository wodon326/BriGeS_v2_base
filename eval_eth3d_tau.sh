set -e
set -x

CUDA_VISIBLE_DEVICES=1 python AsymKD_evaluate_affine_inv_gpu_tau.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir output/eth3d \
    --alignment_max_res 1024