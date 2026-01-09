import torch
from diffusers import SD3Transformer2DModel
from safetensors.torch import load_file, save_file
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Extract residual weights directly from a trained LoRA checkpoint (Inverted ICA Architecture)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base SD3 model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the trained LoRA checkpoint (.safetensors)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted residual weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    return parser.parse_args()

def extract_residual(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"Loading SD3 Transformer from {args.model_path}...")
    try:
        transformer = SD3Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        ).to(device)
    except OSError:
        transformer = SD3Transformer2DModel.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
        ).to(device)

    print(f"Loading LoRA weights from {args.lora_path}...")
    lora_state_dict = load_file(args.lora_path)

    residual_dict = {}
    count = 0
    
    print("Calculating residuals based on Inverted ICA (W ~ S @ A.T)...")
    
    for name, module in transformer.named_modules():
        if not hasattr(module, "weight"):
            continue
            
        # 常见前缀处理
        lora_key_base = f"transformer.{name}" 
        q_key = f"{lora_key_base}.lora.q.weight"       # 现在对应 Mixing Matrix A (Fixed)
        base_r_key = f"{lora_key_base}.lora.base_r.weight" # 现在对应 Base Source S (Fixed)
        
        # 兼容性查找
        if q_key not in lora_state_dict:
            q_key = f"{name}.lora.q.weight"
            base_r_key = f"{name}.lora.base_r.weight"

        if q_key in lora_state_dict and base_r_key in lora_state_dict:
            original_weight = module.weight.data.to(device=device, dtype=dtype)
            
            # --- 关键修改点: 矩阵加载与重构逻辑 ---
            
            # 1. 加载 A_mixing (Fixed Projection)
            # 在 save_model_hook 中，我们将 frozen_A 保存为 q.weight
            # 形状: (In_features, Rank)
            A_mixing = lora_state_dict[q_key].to(device=device, dtype=dtype)
            
            # 2. 加载 S_base (Fixed Source)
            # 在 save_model_hook 中，我们将 lora_S_base 保存为 base_r.weight
            # 形状: (Out_features, Rank)
            S_base = lora_state_dict[base_r_key].to(device=device, dtype=dtype)
            
            # 3. 计算初始重构: W_recon = S_base @ A_mixing.T
            # Dimensions: (Out, Rank) @ (Rank, In) -> (Out, In)
            # 这与标准 Linear 层的 weight 形状一致
            W_recon = torch.mm(S_base, A_mixing.t())
            
            # 处理 Conv2d 或 Fan_in_fan_out
            if getattr(module, "fan_in_fan_out", False):
                W_recon = W_recon.T
            
            if original_weight.dim() == 4:
                W_recon = W_recon.view_as(original_weight)
            
            # 4. 计算残差
            W_res = original_weight - W_recon
            
            res_save_key = f"{name}.residual.weight"
            residual_dict[res_save_key] = W_res.cpu()
            
            count += 1
            
            del original_weight, A_mixing, S_base, W_recon, W_res

    if count == 0:
        print("Warning: No matching layers found! Please check keys.")
    else:
        print(f"Successfully processed {count} layers.")
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "sd3_residual_weights_inverted.safetensors")
        save_file(residual_dict, save_path)
        print(f"Residual weights saved to: {save_path}")

if __name__ == "__main__":
    extract_residual(parse_args())