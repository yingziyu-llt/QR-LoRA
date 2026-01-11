import torch
from diffusers import SD3Transformer2DModel
import sys, os
import argparse
from safetensors.torch import save_file

# 确保能找到你的 train_scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_scripts.custom_svd_lora import inject_svd_lora_layer, SVDLoraLayer

def parse_args():
    parser = argparse.ArgumentParser(description="Save residual weights for SVD-LoRA (Fixed-U) - Memory Efficient")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--rank", type=int, default=64, help="Rank")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()

def save_residual_weights(args):
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device

    print(f"Loading SD3 Transformer ONLY from {args.model_path}...")
    
    # [关键修改]：只加载 Transformer，跳过 T5、CLIP 和 VAE
    try:
        transformer = SD3Transformer2DModel.from_pretrained(
            args.model_path, 
            subfolder="transformer", # 即使是本地文件夹，只要符合 diffusers 格式通常也需要这个
            torch_dtype=dtype
        ).to(device)
    except OSError:
        # 如果用户指的路径里没有 subfolder 结构（比如直接指到了 transformer 文件夹）
        print("Warning: 'transformer' subfolder not found, trying to load directly...")
        transformer = SD3Transformer2DModel.from_pretrained(
            args.model_path, 
            torch_dtype=dtype
        ).to(device)

    # SD3 Target Modules
    target_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out"
    ]

    original_weights = {}
    print("Backing up original weights...")
    for name, module in transformer.named_modules():
        for target in target_modules:
            if name.endswith(target) and hasattr(module, "weight"):
                # 存到 CPU 以节省显存
                original_weights[name] = module.weight.data.cpu().clone()

    print(f"Injecting SVD-LoRA (Rank={args.rank})... This triggers SVD decomposition.")
    
    # 注入层会触发 init_weights，执行 SVD：W ~ U @ M_base
    # 注意：此时 transformer 已经在 GPU 上，SVD 也会在 GPU 上运算（PyTorch GPU SVD 很快）
    transformer = inject_svd_lora_layer(transformer, target_modules=target_modules, rank=args.rank, alpha=args.rank)

    residual_dict = {}
    count = 0
    
    print("Calculating residuals (W_res = W_orig - U @ M_base)...")
    for name, module in transformer.named_modules():
        if isinstance(module, SVDLoraLayer):
            if name not in original_weights:
                continue
            
            # 1. 获取原始权重 (移回 GPU 计算)
            W_orig = original_weights[name].to(device=device, dtype=dtype)
            
            # 2. 获取 SVD 组件
            U_fixed = module.frozen_U["default"].to(device=device, dtype=dtype)
            M_base = module.lora_base_M["default"].to(device=device, dtype=dtype)
            
            # 3. 重构
            W_recon = torch.mm(U_fixed, M_base)
            
            # 4. 处理转置
            if getattr(module, "fan_in_fan_out", False):
                W_recon = W_recon.T
            
            if W_orig.dim() == 4:
                W_recon = W_recon.view_as(W_orig)
                
            # 5. 计算残差
            W_res = W_orig - W_recon
            
            # 存回 CPU
            residual_dict[f"{name}.residual.weight"] = W_res.cpu()
            count += 1
            
            # 及时清理，防止显存碎片
            del W_orig, U_fixed, M_base, W_recon, W_res

    if count == 0:
        print("Warning: No layers processed! Check your target_modules or model path.")
    else:
        print(f"Processed {count} layers.")
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "sd3_svd_residual.safetensors")
        save_file(residual_dict, save_path)
        print(f"Saved residuals to {save_path}")

if __name__ == "__main__":
    save_residual_weights(parse_args())