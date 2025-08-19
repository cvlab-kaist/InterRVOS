#!/bin/bash

export PYTHONPATH=$(pwd)
TARGET_PATH=$1
SAVE_PATH=$2
CONFIG=$3

python model/hf/convert_to_hf.py model/configs/$CONFIG.py \
    --pth-model $TARGET_PATH \
    --save-path $SAVE_PATH \
