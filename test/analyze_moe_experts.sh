python test/analyze_moe_experts.py \
  --model_path "SD3" \
  --lora_path "exps_SD3/0110-125004-<c>-64-MoE-8/pytorch_lora_weights.safetensors" \
  --prompt "a photo of a <c> dog on the beach" \
  --num_experts 8