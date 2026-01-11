#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

# setting your LORA1_PATH LORA2_PATH
LORA1_PATH="exps_SD3/0109-112228-<s>-64/pytorch_lora_weights.safetensors"
LORA2_PATH="exps_SD3/0109-132010-<c>-64/pytorch_lora_weights.safetensors"
LORA1_NAME="sty"
LORA2_NAME="cnt"

OUTPUT_DIR="output_vis/qrlora_sim-$(date +%Y%m%d-%H%M%S)"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    cp $0 $OUTPUT_DIR/vis_script.sh
fi

echo "OUTPUT_DIR: $OUTPUT_DIR"

python test/visualize_qrlora_similarity.py \
    --lora1_path "$LORA1_PATH" \
    --lora2_path "$LORA2_PATH" \
    --lora1_name "$LORA1_NAME" \
    --lora2_name "$LORA2_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --fixed_scale
