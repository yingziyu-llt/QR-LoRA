#!/bin/bash

MODEL_PATH="SD3"
CONTENT_LORA_PATH="exps_SD3/0109-222328-<c>-64/checkpoint-500/pytorch_lora_weights.safetensors"
STYLE_LORA_PATH="exps_SD3/0109-220921-<s>-64/checkpoint-1000/pytorch_lora_weights.safetensors"
RESIDUAL_PATH="sd3_residuals_exact/sd3_svd_residual.safetensors"

CNT_TRIGGER_NAME="<c>"
STY_TRIGGER_NAME="<s>"

PROMPT="a ${CNT_TRIGGER_NAME} dog in ${STY_TRIGGER_NAME} style"

# 提示词
CNT_TRIGGER="<c>"
STY_TRIGGER="<s>"
PROMPT="a ${CNT_TRIGGER} dog in ${STY_TRIGGER} style"

# [核心] 设置权重列表 (逗号分隔)
STYLE_WEIGHTS="0.6,0.8,1.0"
CONTENT_WEIGHTS="0.6,0.8,1.0"

OUTPUT_DIR="outputs_svd_grid/$(date +%m%d-%H%M)"

echo "Generating with SVD-LoRA Merge..."
echo "Style Weights: $STYLE_WEIGHTS"
echo "Content Weights: $CONTENT_WEIGHTS"

python SD3_dir/inference_svd.py \
    --model_path "$MODEL_PATH" \
    --style_lora "$STYLE_LORA_PATH" \
    --content_lora "$CONTENT_LORA_PATH" \
    --residual_path "$RESIDUAL_PATH" \
    --prompt "$PROMPT" \
    --style_scales "$STYLE_WEIGHTS" \
    --content_scales "$CONTENT_WEIGHTS" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42