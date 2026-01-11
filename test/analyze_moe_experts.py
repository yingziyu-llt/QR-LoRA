import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import sys
from safetensors.torch import load_file
from diffusers import StableDiffusion3Pipeline

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_scripts"))
from custom_moe import inject_moe_qr_lora_layer, MoEQRLoraLayer


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MoE Expert Usage")
    parser.add_argument("--model_path", type=str, required=True, help="Base SD3 Model Path")
    parser.add_argument("--lora_path", type=str, required=True, help="Trained MoE-LoRA Checkpoint (.safetensors)")
    parser.add_argument("--prompt", type=str, default="a dog", help="Prompt to analyze")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="analysis_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型
    print(f"Loading model from {args.model_path}...")
    # 1. 加载模型
    print(f"Loading model from {args.model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    )
    
    # 【修改点 1】: 先暂时把 Transformer 搬到 GPU
    # 这样 SVD 分解会利用 CUDA 加速，几秒钟就能跑完所有层
    pipe.transformer.to(args.device) 
    
    # 2. 注入 MoE 层 (此时 SVD 在 GPU 上运行)
    target_modules = ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", 
                      "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out"]
    
    print("Injecting MoE layers (GPU Accelerated)...")
    inject_moe_qr_lora_layer(
        pipe.transformer,
        target_modules=target_modules,
        rank=args.rank,
        num_experts=args.num_experts,
        alpha=args.rank,
        compute_svd=False
    )
    
    # 【修改点 2】: 注入完成、SVD 算完后，再开启省显存模式
    # 这会自动接管设备管理，把不用的层搬回 CPU
    print("Enabling CPU offload...")
    pipe.enable_model_cpu_offload()
    
    # 3. 加载训练好的权重
    print(f"Loading LoRA weights from {args.lora_path}...")
    state_dict = load_file(args.lora_path)
    
    # 将权重加载到模型中
    # 注意：需要处理 key 的匹配，因为 inject 后的 parameter 还是初始化的
    model_dict = pipe.transformer.state_dict()
    # 过滤出匹配的键并加载
    # 这里的关键是我们需要把 state_dict 里的 key 映射回 transformer 的 module
    # 简单的方法是直接 load_lora_weights (如果 diffusers 支持自定义层)
    # 或者手动赋值
    
    params_loaded = 0
    target_dtype = torch.float16

    for name, module in pipe.transformer.named_modules():
        if hasattr(module, "lora_A"): # 识别 MoE 层
            # 构建 safetensors 中的前缀
            prefix = f"transformer.{name}"
            
            # 加载 Q, Base R, Delta R
            if f"{prefix}.lora.q.weight" in state_dict:
                # 【修改点】：全部转为 target_dtype (float16)
                module.frozen_Q["default"].data = state_dict[f"{prefix}.lora.q.weight"].to(args.device, dtype=target_dtype)
                module.lora_base["default"].data = state_dict[f"{prefix}.lora.base_r.weight"].to(args.device, dtype=target_dtype)
                module.lora_A["default"].data = state_dict[f"{prefix}.lora.delta_r.weight"].to(args.device, dtype=target_dtype)
                
                # 加载 Gate (Linear 层必须匹配输入精度)
                module.gate.gate.weight.data = state_dict[f"{prefix}.gate.weight"].to(args.device, dtype=target_dtype)
                if f"{prefix}.gate.bias" in state_dict:
                    module.gate.gate.bias.data = state_dict[f"{prefix}.gate.bias"].to(args.device, dtype=target_dtype)
                
                params_loaded += 1
                
            # 设为 eval 模式
            module.eval()
            
    print(f"Loaded weights for {params_loaded} layers.")

    # 4. 注册 Hook 来捕获 Gate Scores
    expert_activations = {} # Key: Layer Name, Value: Mean Scores [Experts]
    
    def get_activation_hook(name):
        def hook(model, input, output):
            # input[0] is x: [Batch, Seq, Dim]
            x = input[0]
            # 重新计算一次 scores (或者如果能在 forward 里存下来更好，这里为了不改类定义，重算一次)
            # 注意类型匹配
            with torch.no_grad():
                scores = model.gate(x.to(model.gate.gate.weight.dtype)) # [Batch, Seq, Experts]
                # 对 Batch 和 Seq 维度求平均，得到该层每个 Expert 的平均激活度
                mean_scores = scores.mean(dim=[0, 1]).float().cpu().numpy()
                expert_activations[name] = mean_scores
        return hook

    handles = []
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, "gate") and hasattr(module, "lora_A"):
            handles.append(module.register_forward_hook(get_activation_hook(name)))
    print(f"Registered {len(handles)} hooks for expert activation capture.")
    # 5. 运行推理
    print(f"Running inference for prompt: '{args.prompt}'...")
    with torch.no_grad():
        pipe(args.prompt, num_inference_steps=10, guidance_scale=0.0) # 步数少点，GS=0加速
        
    # 6. 移除 Hooks
    for h in handles:
        h.remove()
        
# ... (前面的代码保持不变) ...

    # 7. 可视化
    print("Generating visualization...")
    layer_names = list(expert_activations.keys())
    
    # 【修改点】：直接使用数字索引作为 Y 轴标签
    # 生成从 0 开始的序列号列表
    layer_indices = [str(i) for i in range(len(layer_names))]
    
    data = np.array([expert_activations[n] for n in layer_names]) # [Layers, Experts]
    
    plt.figure(figsize=(10, 20))
    
    # 【修改点】：yticklabels 使用 layer_indices
    # yticklabels='auto' 也是一种选择，但显式传入 indices 可以确保看到具体编号
    ax = sns.heatmap(
        data, 
        yticklabels=layer_indices, 
        xticklabels=[f"Exp {i}" for i in range(args.num_experts)], 
        cmap="viridis", 
        annot=False
    )
    
    plt.title(f"Expert Activation Heatmap\nPrompt: {args.prompt}")
    plt.xlabel("Experts (Low Freq -> High Freq)")
    plt.ylabel("Layer Index (Sequence Order)") # 修改 Y 轴标题为 "Layer Index"
    
    # 【可选优化】：如果层数非常多（例如几百层），标签会挤在一起。
    # 下面这段代码可以让 Y 轴标签每隔 5 个或 10 个显示一次（视层数而定），保持美观。
    # 如果不需要稀疏显示，可以注释掉下面这几行。
    if len(layer_names) > 50:
        for ind, label in enumerate(ax.get_yticklabels()):
            if ind % 5 != 0:  # 每隔 5 个显示一个
                label.set_visible(False)
    
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, "expert_heatmap.png")
    plt.savefig(save_path)
    print(f"Saved heatmap to {save_path}")
    # ================= [新增] 保存量化数据 =================
    # 保存 Usage 矩阵: [Layers, Experts]
    usage_save_path = os.path.join(args.output_dir, "expert_usage_matrix.npy")
    np.save(usage_save_path, data)
    
    # 保存对应的层名称 (为了确保对比时对齐)
    import json
    names_save_path = os.path.join(args.output_dir, "layer_names.json")
    with open(names_save_path, 'w') as f:
        json.dump(layer_names, f)
        
    print(f"Saved quantitative data to:\n - {usage_save_path}\n - {names_save_path}")
    # =====================================================

    # 打印简要统计
    print("\nAverage Expert Usage (Global):")
    global_avg = np.mean(data, axis=0)
    for i, score in enumerate(global_avg):
        print(f"Expert {i}: {score:.4f}")

if __name__ == "__main__":
    main()