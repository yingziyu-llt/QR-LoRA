import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import load_file
import os
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with SD3 ICA-LoRA merging (Inverted Architecture)")
    
    parser.add_argument("--model_path", type=str, required=True, help="Base SD3 model path")
    parser.add_argument("--style_lora_path", type=str, required=True, help="Style LoRA .safetensors")
    parser.add_argument("--content_lora_path", type=str, required=True, help="Content LoRA .safetensors")
    parser.add_argument("--residual_path", type=str, required=True, help="Residual weights .safetensors")
    
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_sd3_merge")
    
    parser.add_argument("--style_scale", type=float, default=1.0)
    parser.add_argument("--content_scale", type=float, default=1.0)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()

def load_weights(path):
    if path.endswith(".safetensors"):
        return load_file(path)
    return torch.load(path, map_location="cpu")

def merge_and_update_weights(
    pipeline, 
    style_weights, 
    content_weights, 
    residual_weights, 
    style_scale, 
    content_scale, 
    device="cuda",
    dtype=torch.float16
):
    print(f"Merging ICA weights (Inverted: Mixing Sources S)...")
    
    for name, param in pipeline.transformer.named_parameters():
        if not name.endswith(".weight"):
            continue
            
        module_name = name.replace("transformer.", "").replace(".weight", "")
        res_key = f"{module_name}.residual.weight"
        
        if res_key not in residual_weights:
            continue
            
        # ICA Keys Mapping:
        # q.weight      -> A_mixing (Fixed Projection)
        # base_r.weight -> S_base   (Fixed Source Base)
        # delta_r.weight-> delta_S  (Trainable Source Delta)
        
        prefix = f"transformer.{module_name}.lora"
        q_key = f"{prefix}.q.weight" 
        base_r_key = f"{prefix}.base_r.weight"
        delta_r_key = f"{prefix}.delta_r.weight"

        if q_key not in style_weights or delta_r_key not in style_weights:
            continue
            
        # Extract matrices (ensure same device and dtype)
        
        # 1. A_mixing (Fixed, Shared) - Shape: (In, Rank)
        # 既然是 Fixed 的，理论上 Style 和 Content 里的这个矩阵应该是一模一样的
        A_mixing = style_weights[q_key].to(device=device, dtype=dtype)
        
        # 2. S_base (Fixed, Shared) - Shape: (Out, Rank)
        S_base = style_weights[base_r_key].to(device=device, dtype=dtype)
        
        # 3. Delta S (Trainable, Unique) - Shape: (Out, Rank)
        dS_style = style_weights[delta_r_key].to(device=device, dtype=dtype)
        dS_content = content_weights[delta_r_key].to(device=device, dtype=dtype)
        
        # 4. Residual - Shape: (Out, In) or (Out, In, k, k)
        W_res = residual_weights[res_key].to(device=device, dtype=dtype)
        
        # --- Merge Logic ---
        # W = (S_base + s1*dS1 + s2*dS2) @ A_mixing.T
        
        # 先在 Source 空间 (Rank space) 进行线性组合
        S_total = S_base + (style_scale * dS_style) + (content_scale * dS_content)
        
        # 再投影回权重空间
        W_recon = torch.mm(S_total, A_mixing.t())
        
        # Handle shapes
        if W_res.dim() == 4:
            W_recon = W_recon.view_as(W_res)
        elif W_recon.shape != W_res.shape:
            W_recon = W_recon.T
            
        # Final Weight Addition
        W_final = W_res + W_recon
        
        # In-place update
        param.data.copy_(W_final)
        
        del A_mixing, S_base, dS_style, dS_content, W_res, S_total, W_recon, W_final

    print("Weights merged and updated successfully.")

def main():
    args = parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    
    print(f"Loading base model: {args.model_path}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload(device=device)
    
    print("Loading LoRAs and Residuals...")
    style_weights = load_weights(args.style_lora_path)
    content_weights = load_weights(args.content_lora_path)
    residual_weights = load_weights(args.residual_path)
    
    merge_and_update_weights(
        pipe,
        style_weights,
        content_weights,
        residual_weights,
        args.style_scale,
        args.content_scale,
        device=device,
        dtype=dtype
    )
    
    print(f"Generating image for prompt: {args.prompt}")
    generator = torch.Generator(device).manual_seed(args.seed)
    
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        guidance_scale=7.0 
    ).images[0]
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_name = f"sd3_ica_inverted_s{args.style_scale}_c{args.content_scale}.png"
    save_path = os.path.join(args.output_dir, save_name)
    image.save(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()