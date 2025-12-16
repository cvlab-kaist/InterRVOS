#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=$(pwd)

PID=$2
N_PID=$3
EXP_NAME=$4

HF_PATH="/mnt/dataset1/woojeong/pretrained_weights/ReVIOSa-4B"
python model/evaluation/ref_vos_eval.py \
    $HF_PATH \
    --dataset INTERRVOS \
    --work_dir EVAL/$EXP_NAME/interrvos \
    --submit \
    --n_pid $N_PID --pid $PID
