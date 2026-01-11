import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import load_file
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with SVD-LoRA Merge (Multi-Weights Support)")
    parser.add_argument("--model_path", type=str, required=True, help="Base SD3 model")
    parser.add_argument("--style_lora", type=str, required=True, help="Style SVD-LoRA path")
    parser.add_argument("--content_lora", type=str, required=True, help="Content SVD-LoRA path")
    parser.add_argument("--residual_path", type=str, required=True, help="Residual weights path")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs_svd")
    
    # [修改] 支持逗号分隔的字符串输入
    parser.add_argument("--style_scales", type=str, default="1.0", help="Comma-separated list of style weights, e.g. '0.8,0.9,1.0'")
    parser.add_argument("--content_scales", type=str, default="1.0", help="Comma-separated list of content weights, e.g. '0.8,0.9,1.0'")
    
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_weights(path, device):
    if path.endswith(".safetensors"):
        return load_file(path, device=device)
    return torch.load(path, map_location=device)

def merge_svd_weights(pipeline, style_weights, content_weights, residual_weights, s_scale, c_scale, device):
    """
    SVD Weight Merging Logic:
    W_new = U_fixed @ (M_base + s_scale * dM_sty + c_scale * dM_cnt) + W_res
    """
    # 筛选出相关的 Key (沿用 QR 命名习惯: q=U, base_r=M_base, delta_r=Delta_M)
    target_keys = [k for k in style_weights.keys() if "lora.q.weight" in k]
    
    for q_key in target_keys:
        base_prefix = q_key.replace(".lora.q.weight", "") 
        base_m_key = f"{base_prefix}.lora.base_r.weight"
        delta_m_key = f"{base_prefix}.lora.delta_r.weight"
        
        module_name_in_model = base_prefix.replace("transformer.", "")
        res_key = f"{module_name_in_model}.residual.weight"
        
        if res_key not in residual_weights:
            continue

        # 1. 提取矩阵 (保持在 GPU 上以加快计算)
        U = style_weights[q_key].to(device) 
        M_base = style_weights[base_m_key].to(device)
        dM_style = style_weights[delta_m_key].to(device)
        dM_content = content_weights[delta_m_key].to(device)
        W_res = residual_weights[res_key].to(device)
        
        # 2. 线性合并系数矩阵 M
        # 这一步是 SVD-LoRA 解耦特性的核心体现
        M_merged = M_base + (s_scale * dM_style) + (c_scale * dM_content)
        
        # 3. 重构权重 W = U @ M_merged + W_res
        W_recon = torch.mm(U, M_merged)
        
        # 4. 形状适配 (Conv2d / Fan_in_fan_out)
        if W_res.dim() == 4:
            W_recon = W_recon.view_as(W_res)
        elif W_recon.shape != W_res.shape:
            W_recon = W_recon.T
            
        W_final = W_recon + W_res
        
        # 5. 覆盖模型权重
        # 寻找对应的子模块
        module = pipeline.transformer
        parts = module_name_in_model.split(".")
        for part in parts:
            module = getattr(module, part)
            
        # In-place update
        module.weight.data.copy_(W_final)
        
        # 清理临时显存
        del U, M_base, dM_style, dM_content, W_res, M_merged, W_recon, W_final

def main():
    args = parse_args()
    
    print(f"Loading Base SD3: {args.model_path}")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    device = args.device if torch.cuda.is_available() else "cpu"
    device_str = args.device if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda"):
        # 尝试从字符串（如 "cuda:0"）中提取 gpu_id（整数）
        try:
            gpu_id = int(device_str.split(":")[-1])
        except ValueError:
            gpu_id = 0 # 默认 fallback 到 0
        
        print(f"Enabling model cpu offload on GPU {gpu_id}...")
        # 关键修改：传入整数 gpu_id，而不是字符串 device
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe.to(device_str)
    
    print("Loading LoRA & Residual Weights...")
    s_weights = load_weights(args.style_lora, device)
    c_weights = load_weights(args.content_lora, device)
    r_weights = load_weights(args.residual_path, device)
    
    # 解析权重列表
    style_scales = [float(x) for x in args.style_scales.split(",")]
    content_scales = [float(x) for x in args.content_scales.split(",")]
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output Directory: {args.output_dir}")
    print(f"Style Weights: {style_scales}")
    print(f"Content Weights: {content_scales}")

    # 遍历生成
    total = len(style_scales) * len(content_scales)
    count = 0
    
    for s_scale in style_scales:
        for c_scale in content_scales:
            count += 1
            print(f"[{count}/{total}] Merging & Generating | Style: {s_scale:.2f}, Content: {c_scale:.2f}")
            
            # 更新模型权重
            merge_svd_weights(
                pipe, s_weights, c_weights, r_weights, 
                s_scale, c_scale, args.device
            )
            
            # 生成图像
            generator = torch.Generator(args.device).manual_seed(args.seed)
            image = pipe(
                args.prompt, 
                num_inference_steps=28, 
                generator=generator,
                guidance_scale=7.0
            ).images[0]
            
            # 保存
            save_name = f"svd_s{s_scale:.2f}_c{c_scale:.2f}.png"
            image.save(os.path.join(args.output_dir, save_name))

    print("Done!")

if __name__ == "__main__":
    main()