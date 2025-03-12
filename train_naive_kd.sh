python train_naive_kd.py \
    --batch_size 2 \
    --num_steps 1000000 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air \
    --save_dir depth_latent4_diff_block_num_residual_channel_split \
    --ckpt BriGeS_checkpoints/BriGeS_Base_best_model.pth \
    --student_ckpt BriGeS_checkpoints/depth_anything_vitb14.pth \
    --train_style trans