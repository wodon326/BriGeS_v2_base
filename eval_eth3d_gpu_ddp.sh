set -e
set -x

python AsymKD_evaluate_affine_inv_gpu_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir output/eth3d \
    --alignment_max_res 1024 \
    --checkpoint_dir checkpoints_new_loss_001_smooth