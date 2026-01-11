#!/bin/bash

# ================= 配置区域 =================

PYTHON_SCRIPT="inference_ica.py"
MODEL_PATH="/path/to/your/SD3_Model"
CONTENT_LORA_PATH="exps_SD3/0109-222328-<c>-64/checkpoint-500/pytorch_lora_weights.safetensors"
STYLE_LORA_PATH="exps_SD3/0109-220921-<s>-64/checkpoint-1000/pytorch_lora_weights.safetensors"
RESIDUAL_PATH="sd3_residuals_exact/sd3_svd_residual.safetensors"
CNT_TRIGGER="<c>"
STY_TRIGGER="<s>"
PROMPT="a ${CNT_TRIGGER} dog in ${STY_TRIGGER} style"
STYLE_WEIGHTS="1.0"
CONTENT_WEIGHTS="0.6"
OUTPUT_DIR="outputs_ica_inverted/$(date +%m%d-%H%M)"

echo "========================================================="
echo "Starting ICA-LoRA (Inverted) Inference"
echo "Model: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Style Weights: $STYLE_WEIGHTS"
echo "Content Weights: $CONTENT_WEIGHTS"
echo "========================================================="

mkdir -p "$OUTPUT_DIR"

python "$PYTHON_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --style_lora_path "$STYLE_LORA_PATH" \
    --content_lora_path "$CONTENT_LORA_PATH" \
    --residual_path "$RESIDUAL_PATH" \
    --prompt "$PROMPT" \
    --style_scale "$STYLE_WEIGHTS" \
    --content_scale "$CONTENT_WEIGHTS" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    --num_inference_steps 28 \
    --dtype "fp16"

echo "========================================================="
echo "All tasks finished. Results saved to $OUTPUT_DIR"