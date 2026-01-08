import torch
from diffusers import StableDiffusion3Pipeline
import sys, os
import argparse
from safetensors.torch import save_file

# 确保能找到 custom_delta_r_lora 等模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 如果脚本在 SD3_dir 下，可能需要 append 上一级
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 根据你训练时使用的方法导入对应的 Injector
# 如果你用的是 ICA，请取消注释 ICA 的部分
from train_scripts.custom_delta_r_lora import inject_delta_r_lora_layer, DeltaRLoraLayer
from train_scripts.custom_delta_r_triu_lora import inject_delta_r_triu_lora_layer, DeltaRTriuLoraLayer
# from train_scripts.custom_ica_lora import inject_ica_lora_layer, ICALoraLayer

def parse_args():
    parser = argparse.ArgumentParser(description="Save residual weights for SD3 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save residual weights")
    parser.add_argument("--rank", type=int, default=64, help="Rank for LoRA")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lora_init_method", type=str, default="triu_deltaR", choices=["deltaR", "triu_deltaR", "ica"])
    return parser.parse_args()

def save_residual_weights(
    pipeline: StableDiffusion3Pipeline,
    output_dir: str,
    rank: int = 64,
    dtype: torch.dtype = torch.float32,
    init_method: str = "triu_deltaR"
) -> None:
    """save residual weights (W_res = W - Q@R)"""
    transformer = pipeline.transformer
    transformer = transformer.to(dtype)
    
    # SD3 的 Target Modules
    target_modules = [
        "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out"
    ]
    
    residual_dict = {}
    original_weights = {}
    
    # 1. 保存原始权重
    for name, module in transformer.named_modules():
        module_parts = name.split(".")
        # 简单匹配逻辑
        for target in target_modules:
            if name.endswith(target):
                if hasattr(module, "weight"):
                    print(f"Found target module: {name}")
                    original_weights[name] = module.weight.data.cpu()

    # 2. 注入 LoRA 层 (触发分解)
    print(f"Injecting LoRA layers with method: {init_method}...")
    if init_method == "deltaR":
        transformer = inject_delta_r_lora_layer(transformer, target_modules=target_modules, rank=rank, alpha=rank)
    elif init_method == "triu_deltaR":
        transformer = inject_delta_r_triu_lora_layer(transformer, target_modules=target_modules, rank=rank, alpha=rank)
    elif init_method == "ica":
        from train_scripts.custom_ica_lora import inject_ica_lora_layer # 延迟导入
        transformer = inject_ica_lora_layer(transformer, target_modules=target_modules, rank=rank, alpha=rank)

    # 3. 计算残差 W_res = W_orig - Q @ R_base
    for name, module in transformer.named_modules():
        # 检查是否是我们替换过的层
        is_our_layer = False
        try:
            if init_method == "ica":
                from train_scripts.custom_ica_lora import ICALoraLayer
                if isinstance(module, ICALoraLayer): is_our_layer = True
            else:
                if isinstance(module, (DeltaRLoraLayer, DeltaRTriuLoraLayer)): is_our_layer = True
        except:
            if isinstance(module, (DeltaRLoraLayer, DeltaRTriuLoraLayer)): is_our_layer = True

        if is_our_layer:
            # 找到对应的原始权重 key
            # 注意：injector 可能会改变层级结构，但通常 name 保持一致
            if name in original_weights:
                original_weight = original_weights[name]
            else:
                print(f"Warning: Could not find original weights for {name}")
                continue
            
            original_weight = original_weight.to(dtype)
            
            # 获取 Q 和 Base R
            if init_method == "ica":
                q_weight = module.frozen_S["default"].to(dtype).cpu() # S 矩阵
                base_r_weight = module.lora_base["default"].to(dtype).cpu() # A.T 矩阵
                # ICA Forward: S @ A.T @ x.T -> W = S @ A.T
                # Q (out, rank), R (rank, in)
                reconstructed = torch.mm(q_weight, base_r_weight)
            else:
                q_weight = module.frozen_Q["default"].to(dtype).cpu()
                base_r_weight = module.lora_base["default"].to(dtype).cpu()
                reconstructed = torch.mm(q_weight, base_r_weight)
            
            # 如果存在 fan_in_fan_out (Conv2d), 权重形状可能需要转置
            if module.fan_in_fan_out:
                reconstructed = reconstructed.T

            w_res = original_weight - reconstructed
            residual_dict[f"{name}.residual.weight"] = w_res

    os.makedirs(output_dir, exist_ok=True)
    save_file(residual_dict, os.path.join(output_dir, "sd3_residual_weights.safetensors"))
    print(f"Residual weights saved to {output_dir}/sd3_residual_weights.safetensors")

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(device)
    
    save_residual_weights(
        pipe,
        args.output_dir,
        args.rank,
        dtype,
        args.lora_init_method
    )

if __name__ == "__main__":
    main()
