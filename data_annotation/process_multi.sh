#!/bin/bash

python visual_prompts/multi_object.py \
    --image_root datasets/VidOR/JPEGImages_clips \
    --mask_root datasets/VidOR/MaskAnnotations \
    --output_dir datasets/VidOR/processed/interaction \
    --vp_type mask_overlay \
    --num_segments 16
