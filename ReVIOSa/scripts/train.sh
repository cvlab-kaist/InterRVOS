#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
CONFIG=$2
N_GPU=$3

bash tools/dist.sh train model/configs/$CONFIG.py $N_GPU
