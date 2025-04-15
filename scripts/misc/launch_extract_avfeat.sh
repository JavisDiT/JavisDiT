#!/bin/bash

set -x
set -e

# ############### prior ###############
# SAVE_PATH=${SAVE_PATH:-"data/feat/prior"}
# CFG_PATH=${CFG_PATH-"configs/javisdit-v0-1/misc/extract_st_prior_va.py"}
# DATA_PATH=${DATA_PATH:-"data/meta/prior/train_prior.csv"}
# #####################################

############### jav ###############
SAVE_PATH=${SAVE_PATH:-"data/feat/jav"}
CFG_PATH=${CFG_PATH-"configs/javisdit-v0-1/misc/extract_va.py"}
DATA_PATH=${DATA_PATH:-"data/meta/TAVGBench/train_jav.csv"}
###################################

NUM_SPLIT=${NUM_SPLIT:-1000}

START_SPLIT=0

DATA_ARG="--data-path $DATA_PATH"
SAVE_ARG="--save-dir $SAVE_PATH"

CMD="torchrun --standalone --nproc_per_node scripts/misc/extract_feat.py $CFG_PATH $DATA_ARG $SAVE_ARG "

declare -a GPUS=(0 1 2 3 4 5 6 7)

mkdir -p logs/extract_feat

for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$gpu $CMD --start-index $(($START_SPLIT + i * $NUM_SPLIT)) --end-index $(($START_SPLIT + (i + 1) * $NUM_SPLIT)) >logs/extract_feat/$gpu.log 2>&1 &
done
