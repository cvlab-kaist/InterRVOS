#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP_NAME=$1

python tools/eval/merge_json.py \
    --result_dir ReVIOSa/EVAL/$EXP_NAME/mevis/valid_u \
    --save_path ReVIOSa/EVAL/$EXP_NAME/mevis/valid_u/results.json

python tools/eval/eval_mevis.py \
    --pred_path ReVIOSa/EVAL/$EXP_NAME/mevis/valid_u/results.json \
    --data_dir datasets/mevis/valid_u \
    --save_path ReVIOSa/EVAL/$EXP_NAME/mevis/valid_u/eval.json
