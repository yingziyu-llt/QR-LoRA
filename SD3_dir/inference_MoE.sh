python ./SD3_dir/inference_moe.py --model_path SD3 \
    --content_lora_path "./exps_SD3/0110-125004-<c>-64-MoE-8/pytorch_lora_weights.safetensors" \
    --style_lora_path "./exps_SD3/0110-170919-<s>-64-MoE-8/pytorch_lora_weights.safetensors" \
    --prompt "a <c> dog in <s> style" \
    --style_scale 1.0 \
    --content_scale 1.0 \
    --rank 64 \
    --num_experts 8 \
    --alpha 64