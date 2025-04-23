#!/bin/bash

NAME=movi_a_test_04_23_gs_color_compete_feature_embedding
DATA_DIR=/home/skyworker/result/4DGS_SlotAttention/shape_of_motion
OUT_DIR=/home/skyworker/result/4DGS_SlotAttention/slot_4dgs

# Set CUDA devices 1 and 2
export CUDA_VISIBLE_DEVICES=1,2,3,4

python train.py \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/$NAME \
    --cfg ./configs/raw_gs.yaml