#!/bin/bash

# model path
MODEL_PATH="./FLUX.1"
RANK=64
OUTPUT_DIR="flux_dir"

export CUDA_VISIBLE_DEVICES=$1

python flux_dir/save_flux_residual.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --rank $RANK 
