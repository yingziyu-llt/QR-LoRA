import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file
import os
import argparse
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with multiple LoRA weights and weighted fusion.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--style_lora_path", type=str, required=True, help="Path to the style LoRA weights.")
    parser.add_argument("--content_lora_path", type=str, required=True, help="Path to the content LoRA weights.")
    parser.add_argument("--residual_path", type=str, required=True, help="Path to the residual weights.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--style_weights", type=str, required=True, help="Comma-separated list of style weights.")
    parser.add_argument("--content_weights", type=str, required=True, help="Comma-separated list of content weights.")
    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp32", help="Data type for model weights.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    
    return parser.parse_args()

def load_weights(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return load_file(path)
    return torch.load(path)

def update_model_weights(
    pipeline: FluxPipeline,
    style_lora_weights: Dict[str, torch.Tensor],
    content_lora_weights: Dict[str, torch.Tensor],
    residual_weights: Dict[str, torch.Tensor],
    style_scale: float,
    content_scale: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> FluxPipeline:
    """W = Q(R + merged_deltaR) + W_res"""
    
    for name, module in pipeline.transformer.named_modules():
        if hasattr(module, "weight"):
            q_key = f"transformer.{name}.lora.q.weight"
            style_r_key = f"transformer.{name}.lora.delta_r.weight"
            content_r_key = f"transformer.{name}.lora.delta_r.weight"
            base_r_key = f"transformer.{name}.lora.base_r.weight"
            residual_key = f"{name}.residual.weight"
            
            if all(k in style_lora_weights for k in [q_key, base_r_key, style_r_key]) and \
            all(k in content_lora_weights for k in [q_key, base_r_key, content_r_key]) and \
            residual_key in residual_weights:
                
                Q = style_lora_weights[q_key].to(device, dtype)
                base_R = style_lora_weights[base_r_key].to(device, dtype)
                style_delta_R = style_lora_weights[style_r_key].to(device, dtype)
                content_delta_R = content_lora_weights[content_r_key].to(device, dtype)
                W_res = residual_weights[residual_key].to(device, dtype)
                
                merged_delta_R = style_scale * style_delta_R + content_scale * content_delta_R
                
                # W = Q(R + merged_deltaR) + W_res
                R_total = base_R + merged_delta_R
                W = torch.mm(Q, R_total) + W_res
                
                module.weight.data = W
                
                del Q, base_R, style_delta_R, content_delta_R, W_res, merged_delta_R, R_total, W
                torch.cuda.empty_cache()

    return pipeline

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(device)
    
    style_lora_weights = load_weights(args.style_lora_path)
    content_lora_weights = load_weights(args.content_lora_path)
    residual_weights = load_weights(args.residual_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    
    style_weights = [float(w) for w in args.style_weights.split(',')]
    content_weights = [float(w) for w in args.content_weights.split(',')]
    
    for style_scale in style_weights:
        for content_scale in content_weights:
            pipe = update_model_weights(
                pipe,
                style_lora_weights,
                content_lora_weights,
                residual_weights,
                style_scale,
                content_scale,
                device,
                dtype,
            )
            
            with torch.autocast(device_type='cuda', dtype=dtype):
                generator = torch.Generator(device=device).manual_seed(args.seed)
                with torch.no_grad():
                    image = pipe(
                        prompt=args.prompt,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                    ).images[0]
            
            image_filename = f"inference_s{style_scale:.2f}_c{content_scale:.2f}.png"
            image.save(os.path.join(args.output_dir, image_filename))
            print(f"Image saved to {os.path.join(args.output_dir, image_filename)}")

if __name__ == "__main__":
    main() 
