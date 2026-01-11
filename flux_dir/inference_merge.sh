#!/bin/bash

MODEL_PATH="./FLUX.1"
CONTENT_LORA_PATH="exps_flux/1202-103526-<c>-64/checkpoint-1000/pytorch_lora_weights.safetensors"
STYLE_LORA_PATH="exps_flux/1202-094302-<s>-64/checkpoint-1000/pytorch_lora_weights.safetensors"
RESIDUAL_PATH="flux_dir/flux_residual_weights.safetensors"

CNT_TRIGGER_NAME="<c>"
STY_TRIGGER_NAME="<s>"

PROMPT="a ${CNT_TRIGGER_NAME} dog in ${STY_TRIGGER_NAME} style"

NUM_STEPS=28
SEED=42
STYLE_WEIGHTS="0.8,0.9,1.0"
CONTENT_WEIGHTS="0.8,0.9,1.0"
DTYPE="fp16"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs_infer_flux/${TIMESTAMP}"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    cp $0 "$OUTPUT_DIR/script.sh"
fi

export CUDA_VISIBLE_DEVICES=$1

python flux_dir/inference_merge_residual.py \
    --model_path "$MODEL_PATH" \
    --style_lora_path "$STYLE_LORA_PATH" \
    --content_lora_path "$CONTENT_LORA_PATH" \
    --residual_path "$RESIDUAL_PATH" \
    --prompt "$PROMPT" \
    --num_inference_steps $NUM_STEPS \
    --seed $SEED \
    --style_weights "$STYLE_WEIGHTS" \
    --content_weights "$CONTENT_WEIGHTS" \
    --dtype "$DTYPE" \
    --output_dir "$OUTPUT_DIR"
