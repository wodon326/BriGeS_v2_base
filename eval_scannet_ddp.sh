set -e
set -x

python AsymKD_evaluate_affine_inv_ddp.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir output/scannet \
    --checkpoint_dir checkpoints_new_loss_001_smooth
