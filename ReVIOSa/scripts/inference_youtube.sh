#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=$(pwd)

PID=$2
N_PID=$3
EXP_NAME=$4

HF_PATH="wooj0216/ReVIOSa-4B"
python model/evaluation/ref_vos_eval.py \
    $HF_PATH \
    --submit \
    --dataset REFYTVOS \
    --work_dir EVAL/$EXP_NAME/ref_youtube \
    --n_pid $N_PID --pid $PID
