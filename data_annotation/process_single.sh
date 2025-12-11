#!/bin/bash

python visual_prompts/single_object.py \
    --image_root datasets/VidOR/JPEGImages_clips \
    --mask_root datasets/VidOR/MaskAnnotations \
    --output_dir datasets/VidOR/processed/single \
    --num_segments 16
