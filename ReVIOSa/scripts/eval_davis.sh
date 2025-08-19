#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP_NAME=$1

python tools/eval/merge_json.py \
    --result_dir ReVIOSa/EVAL/$EXP_NAME/ref_davis \
    --save_path ReVIOSa/EVAL/$EXP_NAME/ref_davis/results.json

python tools/eval/eval_mevis.py \
    --pred_path ReVIOSa/EVAL/$EXP_NAME/ref_davis/results.json \
    --data_dir datasets/ref_davis/valid \
    --save_path ReVIOSa/EVAL/$EXP_NAME/ref_davis/eval.json
