#!/bin/bash

NAME=movi_a_test_05_06_split_l1_l2_fg_gs
DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs

# Set CUDA devices 2 and 3
export CUDA_VISIBLE_DEVICES=0,1

python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/$NAME \
    --cfg ./configs/raw_gs.yaml