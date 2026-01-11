python test/analyze_moe_experts.py \
  --model_path "SD3" \
  --lora_path "exps_SD3/0110-170919-<s>-64-MoE-8/pytorch_lora_weights.safetensors" \
  --prompt "a cat in <s> style" \
  --num_experts 8