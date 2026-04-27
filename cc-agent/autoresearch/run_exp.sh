#!/bin/bash
# SPAGS Autoresearch experiment runner
# Usage: CUDA_VISIBLE_DEVICES=X run_exp.sh <output_name> [extra_args]

SPAGS_DIR="/home/qyhu/SPAGS"
PYTHON="/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python"
DATA="data/spags_format/foot_50_3views"
NAME="$1"
shift 1

cd "$SPAGS_DIR" || exit 1

$PYTHON train.py \
  -s "$DATA" \
  -m "output/autoresearch/$NAME" \
  --iterations 3000 --test_iterations 1000 2000 3000 \
  --eval "$@" \
  > "output/autoresearch/${NAME}.log" 2>&1
