#!/bin/bash
cd /home/qyhu/SPAGS || exit 1
export CUDA_VISIBLE_DEVICES=$1
shift
exec /home/qyhu/anaconda3/envs/cu116torch112_1/bin/python train.py "$@"
