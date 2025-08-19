#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP_NAME=$1

python tools/eval/merge_json.py \
    --result_dir ReVIOSa/EVAL/$EXP_NAME/interrvos \
    --save_path ReVIOSa/EVAL/$EXP_NAME/interrvos/results.json

python tools/eval/eval_interrvos.py \
    --pred_path ReVIOSa/EVAL/$EXP_NAME/interrvos/results.json \
    --data_dir datasets/InterRVOS/val \
    --save_path ReVIOSa/EVAL/$EXP_NAME/interrvos/eval_actor.json
