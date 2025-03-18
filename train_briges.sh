python train_briges.py \
    --batch_size 4 \
    --num_steps 1000000 \
    --lr 0.00005 \
    --train_datasets HRWSI BlendedMVS tartan_air \
    --save_dir BriGeS_decoder_unfreeze \
    --train_style trans