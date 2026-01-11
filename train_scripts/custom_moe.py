import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class MoEGate(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=None):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)
        
        if self.top_k is not None:
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, top_k_indices, 1.0)
            scores = scores * mask
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
            
        return scores

class MoEQRLoraLayer(nn.Module):
    def __init__(
        self,
        base_layer,
        rank: int = 64,
        num_experts: int = 4,
        alpha: int = 64,
        delta_scale: float = 0.01,
        init_method: str = "moe_split",
        use_zero_init: bool = True,
        compute_svd: bool = True,
        **kwargs
    ):
        super().__init__()
        self.base_layer = base_layer
        self.init_method = init_method
        self.use_zero_init = use_zero_init
        self.num_experts = num_experts
        self.rank = rank
        self.compute_svd = compute_svd
        
        assert rank % num_experts == 0, f"Rank ({rank}) must be divisible by num_experts ({num_experts})"
        self.expert_rank = rank // num_experts
        
        self.r = {"default": rank}
        self.scaling = {"default": alpha / rank}
        
        if hasattr(base_layer, "in_features"):
            self.in_features = base_layer.in_features
        else:
            self.in_features = base_layer.in_channels
        if hasattr(base_layer, "out_features"):
            self.out_features = base_layer.out_features
        else:
            self.out_features = base_layer.out_channels
            
        self.fan_in_fan_out = getattr(base_layer, "fan_in_fan_out", False)
        
        # 1. 门控网络
        self.gate = MoEGate(self.in_features, num_experts)
        
        # 2. 参数存储：直接存大矩阵，不再分块存储
        # lora_A (Delta R): [rank, in_features]
        # lora_base (R): [rank, in_features]
        # frozen_Q (Q): [out_features, rank]
        
        self.lora_A = nn.ParameterDict({}) 
        self.lora_base = nn.ParameterDict({})
        self.frozen_Q = {}
        
        self.init_weights("default", delta_scale)

    def init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        # 初始化占位符
        device = self.base_layer.weight.device
        
        # Q: [Out, Rank]
        self.frozen_Q[adapter_name] = torch.zeros(
            self.out_features, self.rank, device=device, dtype=torch.float32
        )
        # Base R: [Rank, In]
        self.lora_base[adapter_name] = nn.Parameter(
            torch.zeros(self.rank, self.in_features, device=device, dtype=torch.float32),
            requires_grad=False
        )
        # Delta R: [Rank, In]
        self.lora_A[adapter_name] = nn.Parameter(
            torch.zeros(self.rank, self.in_features, device=device, dtype=torch.float32)
        )
        
        # 执行分组初始化并填充大矩阵
        self.grouped_svd_init_fast(adapter_name, delta_scale)

    def grouped_svd_init_fast(self, adapter_name: str, delta_scale: float):
        base_weights = self.base_layer.weight.data
        device = base_weights.device
        original_dtype = base_weights.dtype
        
        if self.fan_in_fan_out:
            base_weights = base_weights.T
            
        try:
            if not self.compute_svd:
                raise RuntimeError("Skipping SVD computation as per configuration.")
            # 1. 全局 SVD
            base_weights_fp32 = base_weights.to(torch.float32)
            U, S, Vh = torch.linalg.svd(base_weights_fp32, full_matrices=False)
            
            # 用于累加重建的权重，从原始权重中减去
            total_recon = torch.zeros_like(base_weights_fp32)
            
            # 2. 循环处理分组，填入大矩阵的对应切片位置
            for i in range(self.num_experts):
                start = i * self.expert_rank
                end = (i + 1) * self.expert_rank
                
                # SVD Slice
                u_slice = U[:, start:end]
                # 注意：这里需要先处理 scaling
                s_slice = S[start:end] / self.scaling[adapter_name]
                vh_slice = Vh[start:end, :]
                
                # Core Matrix (Rank = expert_rank)
                core_matrix = (u_slice @ torch.diag(s_slice)) @ vh_slice
                
                # QR Decomposition
                Q_i, R_i = torch.linalg.qr(core_matrix, mode='reduced')
                
                # [关键修复]：截取前 expert_rank 个分量
                # Q_i shape: [out, min(out, in)] -> 截取 [out, expert_rank]
                # R_i shape: [min(out, in), in] -> 截取 [expert_rank, in]
                Q_sub = Q_i[:, :self.expert_rank]
                R_sub = R_i[:self.expert_rank, :]
                
                # 填入大矩阵 (In-place copy)
                self.frozen_Q[adapter_name][:, start:end] = Q_sub.detach()
                self.lora_base[adapter_name].data[start:end, :] = R_sub.detach()
                
                # lora_A (Delta R) 初始化
                if not self.use_zero_init:
                    self.lora_A[adapter_name].data[start:end, :] = torch.randn_like(R_sub) * delta_scale
                else:
                    self.lora_A[adapter_name].data[start:end, :] = torch.zeros_like(R_sub)
                
                # 累加重建部分 (使用截取后的矩阵)
                total_recon += (self.scaling[adapter_name] * (Q_sub @ R_sub))
            
            # 3. 更新 Base Weights
            base_weights_fp32 -= total_recon
            
            # 清理显存
            del U, S, Vh, total_recon, Q_i, R_i, Q_sub, R_sub, core_matrix
            torch.cuda.empty_cache()
            
            # 回写
            weight = base_weights_fp32.to(original_dtype)
            if self.fan_in_fan_out:
                weight = weight.T
            self.base_layer.weight.data = weight
            
        except RuntimeError as e:
            # print(f"SVD init failed: {e}, utilizing random initialization.")
            # Fallback (Standard LoRA-like init)
            self.frozen_Q[adapter_name] = torch.randn(self.out_features, self.rank, device=device)
            self.lora_A[adapter_name].data = torch.zeros(self.rank, self.in_features, device=device)
            self.lora_base[adapter_name].data = torch.zeros(self.rank, self.in_features, device=device)
            # 注意：Random init 这种情况下通常不需要从 base_weight 减去什么，
            # 或者你需要手动构造一个随机的 orthogonal 初始化
    def forward(self, x: torch.Tensor):
        # x: [Batch, In]
        dtype = x.dtype
        adapter_name = "default"
        
        # 1. 原始路径
        out = self.base_layer(x)
        
        # 2. 门控计算
        # scores: [Batch, Experts]
        scores = self.gate(x.to(dtype))
        
        # 3. 快速并行计算
        # 准备大矩阵
        Q_all = self.frozen_Q[adapter_name].to(dtype)    # [Out, Rank]
        R_base = self.lora_base[adapter_name].to(dtype)  # [Rank, In]
        R_delta = self.lora_A[adapter_name].to(dtype)    # [Rank, In]
        
        # Step A: 投影到 Rank 空间
        # projected_base:  [Batch, Rank] = x @ R_base.T
        # projected_delta: [Batch, Rank] = x @ R_delta.T
        projected_base = F.linear(x, R_base)
        projected_delta = F.linear(x, R_delta)
        
        # Step B: 应用门控 (关键优化点)
        # 我们需要将 [Batch, Experts] 的 scores 扩展到 [Batch, Rank]
        # 假设 Rank=64, Experts=4, 则每个 expert 控制 16 个通道
        # scores_expanded: [Batch, Rank]
        scores_expanded = scores.repeat_interleave(self.expert_rank, dim=-1)
        
        # 融合：Base 部分不被 Gate 抑制 (保持预训练知识)，Delta 部分被动态加权
        # Rank_Space_Feature = (x @ R) + g(x) * (x @ dR)
        rank_space_feature = projected_base + (projected_delta * scores_expanded)
        
        # Step C: 投影回输出空间
        # [Batch, Rank] @ [Rank, Out] -> [Batch, Out]
        moe_out = F.linear(rank_space_feature, Q_all)
        
        # 4. 最终相加
        return out + moe_out * self.scaling[adapter_name]

def inject_moe_qr_lora_layer(
    model: nn.Module, 
    target_modules: list[str],
    rank: int = 64,
    num_experts: int = 4,
    alpha: int = 64,
    **kwargs
):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                new_layer = MoEQRLoraLayer(
                    module,
                    rank=rank,
                    num_experts=num_experts,
                    alpha=alpha,
                    **kwargs
                )
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_layer)
    return model