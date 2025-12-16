#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

EXP_NAME=$1

python tools/eval/merge_json.py \
    --result_dir EVAL/$EXP_NAME/interrvos/target \
    --save_path EVAL/$EXP_NAME/interrvos/target/results.json

python tools/eval/eval_interrvos_target.py \
    --pred_path EVAL/$EXP_NAME/interrvos/target/results.json \
    --data_dir datasets/InterRVOS/val \
    --save_path EVAL/$EXP_NAME/interrvos/target/eval.json
