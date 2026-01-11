#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
RANK=$2

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

TRIGGER_NAME="<s>"
INSTANCE_DIR="assets/sty/cat"
INSTANCE_PROMPT="a cat in ${TRIGGER_NAME} style"
VALID_PROMPT="a dog in ${TRIGGER_NAME} style"
OUTPUT_DIR="./exps_flux/$(date +%m%d-%H%M%S)-${TRIGGER_NAME}-${RANK}-QR"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
    cp $0 $OUTPUT_DIR/train_script.sh
fi

MODEL_NAME="./FLUX.1"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "NUM_GPUS: $NUM_GPUS"
echo "TRIGGER_NAME: $TRIGGER_NAME"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "INSTANCE_DIR: $INSTANCE_DIR"
echo "MODEL_NAME: $MODEL_NAME"
echo "LoRA_RANK: $RANK"
echo "INSTANCE_PROMPT: $INSTANCE_PROMPT"
echo "VALID_PROMPT: $VALID_PROMPT"

accelerate launch --num_processes=$NUM_GPUS train_scripts/train_qrlora_flux_QR.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --rank=$RANK \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --seed="0" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="$VALID_PROMPT" \
  --validation_epochs=100 \
  --lora_init_method="qr" \
  --checkpointing_steps=250
