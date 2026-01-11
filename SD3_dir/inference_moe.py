import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from diffusers import StableDiffusion3Pipeline
import argparse
import os
import sys

# 包装器：持有参数，解决 ModuleDict 不能存 Tensor 的问题
class MoEAdapterWrapper(nn.Module):
    def __init__(self, gate, Q, dR):
        super().__init__()
        self.gate = gate
        # Q 和 dR 是 nn.Parameter，会自动注册
        self.Q = Q
        self.dR = dR

class MoEMergeInferenceLayer(nn.Module):
    def __init__(self, base_layer, num_experts, rank, alpha):
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = num_experts
        self.expert_rank = rank // num_experts
        self.scaling = alpha / rank
        
        self.adapters = nn.ModuleDict({})
        self.adapter_scales = {} 

    def add_adapter(self, adapter_name, state_dict, prefix):
        # 【关键修复】获取 Base Model 的精度 (通常是 fp16)
        target_dtype = self.base_layer.weight.dtype

        # 1. 加载 Gate
        gate = nn.Linear(self.base_layer.in_features, self.num_experts)
        
        # 【关键修复】加载时强制转为 target_dtype
        w = state_dict[f"{prefix}.gate.weight"].to(dtype=target_dtype)
        gate.weight.data = w.cpu()
        
        if f"{prefix}.gate.bias" in state_dict:
            b = state_dict[f"{prefix}.gate.bias"].to(dtype=target_dtype)
            gate.bias.data = b.cpu()
        
        # 确保 Linear 层本身的属性也是正确的 dtype
        gate.to(dtype=target_dtype)
        
        # 2. 加载 Q 和 Delta R (同样强制转换)
        q_tensor = state_dict[f"{prefix}.lora.q.weight"].to(dtype=target_dtype)
        dr_tensor = state_dict[f"{prefix}.lora.delta_r.weight"].to(dtype=target_dtype)
        
        Q = nn.Parameter(q_tensor.cpu(), requires_grad=False)
        dR = nn.Parameter(dr_tensor.cpu(), requires_grad=False)
            
        # 3. 包装并存入
        self.adapters[adapter_name] = MoEAdapterWrapper(gate, Q, dR)
        self.adapter_scales[adapter_name] = 1.0 

    def set_scale(self, adapter_name, scale):
        self.adapter_scales[adapter_name] = scale

    def forward(self, x):
        # 1. 原始路径
        out = self.base_layer(x)
        dtype = x.dtype
        
        # 2. 遍历所有 Adapter
        for name, module_wrapper in self.adapters.items():
            scale = self.adapter_scales[name]
            if scale == 0:
                continue
            
            # 取出参数
            gate = module_wrapper.gate
            Q_all = module_wrapper.Q
            R_delta = module_wrapper.dR
            
            # --- MoE Delta 计算 ---
            
            # A. 门控 (x 已经是 dtype，gate 权重现在也是 dtype 了)
            scores = F.softmax(gate(x), dim=-1)
            
            # B. 扩展门控
            scores_expanded = scores.repeat_interleave(self.expert_rank, dim=-1)
            
            # C. Delta 投影
            # 显式确保 dtype 一致，虽然上面已经转过了，双重保险
            projected_delta = F.linear(x, R_delta.to(dtype))
            
            # D. 加权
            weighted_delta = projected_delta * scores_expanded
            
            # E. 投影回输出空间
            adapter_out = F.linear(weighted_delta, Q_all.to(dtype))
            
            # F. 累加
            out = out + adapter_out * self.scaling * scale
            
        return out

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with MoE-QR-LoRA merging")
    parser.add_argument("--model_path", type=str, required=True, help="Base SD3 model path")
    parser.add_argument("--style_lora_path", type=str, required=True, help="Style MoE-LoRA path")
    parser.add_argument("--content_lora_path", type=str, required=True, help="Content MoE-LoRA path")
    
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_moe_merge")
    
    parser.add_argument("--style_scale", type=float, default=1.0)
    parser.add_argument("--content_scale", type=float, default=1.0)
    
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=64)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    
    print(f"Loading base model: {args.model_path} with {dtype}")
    # 保持在 CPU，让 offload 管理
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype
    )
    
    print("Enabling model cpu offload...")
    pipe.enable_model_cpu_offload()
    
    adapters = [
        {"name": "content", "path": args.content_lora_path, "scale": args.content_scale},
        {"name": "style",   "path": args.style_lora_path,   "scale": args.style_scale}
    ]
    
    print("Loading LoRA weights and injecting Merge Layers...")
    state_dicts = {cfg["name"]: load_file(cfg["path"]) for cfg in adapters}
    
    count = 0
    # 遍历并替换模型层 (此时模型在 CPU)
    for name, module in pipe.transformer.named_modules():
        first_lora = adapters[0]["name"]
        gate_key = f"transformer.{name}.gate.weight"
        
        if gate_key in state_dicts[first_lora]:
            # 创建融合层
            merge_layer = MoEMergeInferenceLayer(module, args.num_experts, args.rank, args.alpha)
            
            # 加载 Content 和 Style
            for cfg in adapters:
                merge_layer.add_adapter(
                    cfg["name"], 
                    state_dicts[cfg["name"]], 
                    prefix=f"transformer.{name}"
                )
                merge_layer.set_scale(cfg["name"], cfg["scale"])
            
            # 替换
            parent_name, child_name = name.rsplit(".", 1)
            parent = pipe.transformer.get_submodule(parent_name)
            setattr(parent, child_name, merge_layer)
            count += 1
            
    print(f"Injected MoE Merge Layer to {count} modules.")
    
    print(f"Generating image for prompt: {args.prompt}")
    generator = torch.Generator("cpu").manual_seed(args.seed)
    
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        guidance_scale=7.0 
    ).images[0]
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_name = f"moe_merge_s{args.style_scale}_c{args.content_scale}.png"
    save_path = os.path.join(args.output_dir, save_name)
    image.save(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()