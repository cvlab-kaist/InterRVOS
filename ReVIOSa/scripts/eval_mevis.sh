#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP_NAME=$1

python tools/eval/merge_json.py \
    --result_dir EVAL/$EXP_NAME/mevis/valid_u \
    --save_path EVAL/$EXP_NAME/mevis/valid_u/results.json

python tools/eval/eval_mevis.py \
    --pred_path EVAL/$EXP_NAME/mevis/valid_u/results.json \
    --data_dir datasets/mevis/valid_u \
    --save_path EVAL/$EXP_NAME/mevis/valid_u/eval.json
