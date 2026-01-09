import torch
from diffusers import StableDiffusion3Pipeline
import sys, os
import argparse
import random
import numpy as np
from safetensors.torch import save_file

# 添加路径以导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_scripts.custom_delta_r_lora import inject_delta_r_lora_layer, DeltaRLoraLayer
from train_scripts.custom_delta_r_triu_lora import inject_delta_r_triu_lora_layer, DeltaRTriuLoraLayer
from train_scripts.custom_ica_lora import inject_ica_lora_layer, ICALoraLayer

def parse_args():
    parser = argparse.ArgumentParser(description="Save residual weights for SD3 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save residual weights")
    parser.add_argument("--rank", type=int, default=64, help="Rank for LoRA")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lora_init_method", type=str, default="triu_deltaR", choices=["deltaR", "triu_deltaR", "ica"])
    # [新增] 种子参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic initialization (Crucial for ICA)")
    # [新增] 设备参数
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g., 'cuda', 'cuda:0', 'cpu'")
    return parser.parse_args()

def setup_seed(seed):
    """固定所有随机数种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set to: {seed}")

def save_residual_weights(
    pipeline: StableDiffusion3Pipeline,
    output_dir: str,
    rank: int = 64,
    dtype: torch.dtype = torch.float32,
    init_method: str = "triu_deltaR",
    device: str = "cuda"
) -> None:
    """
    Save residual weights:
    - DeltaR/QR: W_res = W_orig - Q @ R_base
    - ICA:       W_res = W_orig - S @ A_base^T
    """
    transformer = pipeline.transformer
    transformer = transformer.to(dtype)
    
    # SD3 Target Modules
    target_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out"
    ]
    
    residual_dict = {}
    original_weights = {}
    
    print(f"1. Loading original weights...")
    for name, module in transformer.named_modules():
        for target in target_modules:
            if name.endswith(target) and hasattr(module, "weight"):
                # 拷贝到 CPU 避免显存占用，后续计算再按需移回
                original_weights[name] = module.weight.data.cpu().clone()

    print(f"2. Injecting LoRA layers ({init_method})...")
    # 注入过程会触发初始化（ICA分解或QR分解），此时依赖全局随机种子
    if init_method == "deltaR":
        transformer = inject_delta_r_lora_layer(transformer, target_modules=target_modules, rank=rank, alpha=rank)
    elif init_method == "triu_deltaR":
        transformer = inject_delta_r_triu_lora_layer(transformer, target_modules=target_modules, rank=rank, alpha=rank)
    elif init_method == "ica":
        transformer = inject_ica_lora_layer(transformer, target_modules=target_modules, rank=rank, alpha=rank)

    print(f"3. Calculating residuals...")
    count = 0
    for name, module in transformer.named_modules():
        is_custom = isinstance(module, (DeltaRLoraLayer, DeltaRTriuLoraLayer, ICALoraLayer))
        
        if is_custom:
            if name not in original_weights:
                print(f"Warning: Original weight not found for {name}")
                continue
            
            # 放到指定设备计算（通常 GPU 更快，但显存不够可以改回 cpu）
            calc_device = device 
            original_weight = original_weights[name].to(dtype=dtype, device=calc_device)
            
            # Calculate Reconstruction based on method
            if isinstance(module, ICALoraLayer):
                # ICA: W ~ S @ A.T
                s_matrix = module.frozen_S["default"].to(dtype=dtype, device=calc_device)
                a_base_t = module.lora_base["default"].to(dtype=dtype, device=calc_device)
                reconstructed = torch.mm(s_matrix, a_base_t)
                
            else:
                # DeltaR: W ~ Q @ R
                if isinstance(module, DeltaRTriuLoraLayer):
                    q_matrix = module.lora_B["default"].to(dtype=dtype, device=calc_device)
                else:
                    q_matrix = module.frozen_Q["default"].to(dtype=dtype, device=calc_device)
                    
                r_base = module.lora_base["default"].to(dtype=dtype, device=calc_device)
                reconstructed = torch.mm(q_matrix, r_base)

            # Handle Transpose for Conv2d or fan_in_fan_out layers
            if getattr(module, "fan_in_fan_out", False):
                reconstructed = reconstructed.T
                
            if original_weight.dim() == 4:
                reconstructed = reconstructed.view_as(original_weight)

            # Residual = Original - Reconstructed
            w_res = original_weight - reconstructed
            
            # 存回 CPU
            residual_dict[f"{name}.residual.weight"] = w_res.cpu()
            count += 1
            
            # 及时释放显存
            del original_weight, reconstructed, w_res
            # torch.cuda.empty_cache() # 频繁调用可能影响速度，可按需开启

    print(f"Processed {count} layers.")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "sd3_residual_weights.safetensors")
    save_file(residual_dict, save_path)
    print(f"Residual weights saved to {save_path}")

def main():
    args = parse_args()
    
    # [新增] 设置种子
    setup_seed(args.seed)
    
    # [修改] 使用指定的 device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        device = "cpu"
    else:
        device = args.device
    
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    
    print(f"Loading SD3 model from {args.model_path} to {device}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(device)
    
    save_residual_weights(
        pipe,
        args.output_dir,
        args.rank,
        dtype,
        args.lora_init_method,
        device # 传入 device
    )

if __name__ == "__main__":
    main()