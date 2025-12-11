#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python run.py \
    --stage gpt \
    --mask_anno_dir datasets/VidOR/MaskAnnotationsV4 \
    --processed_dir datasets/VidOR/processed \
    --output_dir datasets/VidOR/FINAL \
    --num_gpus 4 \
