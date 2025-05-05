#!/bin/bash

NAME=movi_a_test_05_05_fix_padding_test
DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs

# Set CUDA devices 2 and 3
export CUDA_VISIBLE_DEVICES=8

python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/$NAME \
    --cfg ./configs/raw_gs.yaml