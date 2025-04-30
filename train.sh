#!/bin/bash

NAME=movi_a_test_04_30_pos_col_encode_decode_norm_embed_2
DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs

# Set CUDA devices 1 and 2
export CUDA_VISIBLE_DEVICES=2,3

python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/$NAME \
    --cfg ./configs/raw_gs.yaml