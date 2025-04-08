#!/bin/bash

NAME=movi_a_test_0408_rawGs_xyz_col_15slots
DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs

python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/$NAME \
    --cfg ./configs/15_slots.yaml