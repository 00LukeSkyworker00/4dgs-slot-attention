#!/bin/bash

NAME=movi_a_test_04_15_rasterization_4
DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs

python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/$NAME \
    --cfg ./configs/raw_gs.yaml