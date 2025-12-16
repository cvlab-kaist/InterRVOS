#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP_NAME=$1

python tools/eval/merge_json.py \
    --result_dir EVAL/$EXP_NAME/ref_davis \
    --save_path EVAL/$EXP_NAME/ref_davis/results.json

python tools/eval/eval_mevis.py \
    --pred_path EVAL/$EXP_NAME/ref_davis/results.json \
    --data_dir datasets/ref_davis/valid \
    --save_path EVAL/$EXP_NAME/ref_davis/eval.json
