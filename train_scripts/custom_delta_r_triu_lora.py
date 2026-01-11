import torch
import torch.nn as nn
from typing import Union

class DeltaRTriuLoraLayer(nn.Module):
    def __init__(
        self,
        base_layer,
        rank: int = 64,
        alpha: int = 64,
        delta_scale: float = 0.01,
        init_method: str = "triu_deltaR",
        use_zero_init: bool = False,
        **kwargs
    ):
        super().__init__()
        self.base_layer = base_layer
        self.init_method = init_method
        self.use_zero_init = use_zero_init
        
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
        
        self.register_buffer('triu_mask', torch.triu(torch.ones(rank, self.in_features)))
        
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({}) 
        self.lora_base = nn.ParameterDict({})
        self.frozen_Q = {} 
        
        self.init_weights("default", delta_scale)

    def enforce_triangular(self):
        with torch.no_grad():
            for adapter_name in self.lora_A.keys():
                self.triu_mask = self.triu_mask.to(self.lora_A[adapter_name].device)
                self.lora_A[adapter_name].data = self.lora_A[adapter_name] * self.triu_mask
               

    def init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        if self.init_method == "triu_deltaR":
            self.deltaR_init_weights(adapter_name, delta_scale)
        else:
            raise NotImplementedError(f"Init method {self.init_method} not implemented")
        
    @torch.no_grad()
    def deltaR_init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        # 获取原始权重引用
        base_weight_tensor = self.base_layer.weight
        device = base_weight_tensor.device
        dtype = base_weight_tensor.dtype
        rank = self.r[adapter_name]
        
        # 确保 mask 在正确的设备上
        self.triu_mask = self.triu_mask.to(device)
        
        # 1. 准备用于 SVD 的 float32 数据
        # 为了节省显存，我们尽量不保留 W_f32 的长期副本，SVD 后立即释放
        W_f32 = base_weight_tensor.to(device=device, dtype=torch.float32)
        
        if self.fan_in_fan_out:
            W_f32 = W_f32.T
            
        try:
            # 2. SVD 分解 (保持在 GPU)
            # 使用 full_matrices=False 以减少显存
            U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            
            # 截取所需的秩，并立即释放不需要的部分
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            
            # 关键优化：SVD 结束后立即释放大矩阵内存
            del U, S, Vh, W_f32
            torch.cuda.empty_cache()
            
            # 应用缩放
            S_r = S_r / self.scaling[adapter_name]
            
            # 3. 优化的 QR 分解 (Factored QR)
            # 原始逻辑是: QR(U_r @ diag(S_r) @ Vh_r)
            # 为了避免构建巨大的 [out, in] 矩阵，我们分两步做 QR：
            # 令 A = U_r * S_r, B = Vh_r
            # 目标是 QR(A @ B)
            
            # A: [out, rank], B: [rank, in]
            A = U_r * S_r.unsqueeze(0) 
            B = Vh_r
            
            # Step 3.1: 对 A 进行 QR -> Q_a [out, rank], R_a [rank, rank]
            Q_a, R_a = torch.linalg.qr(A, mode='reduced')
            del A, U_r, S_r # 释放
            
            # Step 3.2: 计算中间小矩阵 Temp = R_a @ B -> [rank, in]
            # 这个矩阵非常小，相比于 [out, in] 忽略不计
            Temp = R_a @ B
            del R_a, B, Vh_r # 释放
            
            # Step 3.3: 对 Temp 进行 QR -> Q_b [rank, rank], R_final [rank, in]
            Q_b, R_final = torch.linalg.qr(Temp, mode='reduced')
            del Temp
            
            # Step 3.4: 最终的 Q = Q_a @ Q_b -> [out, rank]
            Q_final = Q_a @ Q_b
            del Q_a, Q_b
            
            # 此时我们得到了 Q_final 和 R_final，且从未构建过完整的 dense matrix
            torch.cuda.empty_cache()
            
            # 4. 保存 LoRA 参数
            self.frozen_Q[adapter_name] = Q_final.detach()
            self.lora_B[adapter_name] = nn.Parameter(Q_final.detach(), requires_grad=False)
            self.lora_base[adapter_name] = nn.Parameter(R_final.detach(), requires_grad=False)
            
            if self.use_zero_init:
                self.lora_A[adapter_name] = nn.Parameter(
                    torch.zeros(rank, self.in_features, device=device, dtype=dtype)
                )
            else:
                delta_R = torch.randn(rank, self.in_features, device=device, dtype=dtype) * delta_scale
                self.lora_A[adapter_name] = nn.Parameter(delta_R * self.triu_mask)
            
            # 5. 更新原始权重 (分块计算以防 OOM)
            # 我们需要计算 W_new = W_old - scale * (Q_final @ R_final)
            # 为了避免分配 (Q_final @ R_final) 这一巨大矩阵，我们分块计算并更新
            
            # 准备计算用的 float32 副本
            Q_calc = Q_final.float()
            R_calc = R_final.float()
            scale = self.scaling[adapter_name]
            
            # 确定物理存储的维度方向
            # 如果 fan_in_fan_out=True, 物理存储是 [in, out], 我们逻辑计算的是 [out, in]
            # 这种情况下，我们需要减去 (Q @ R).T = R.T @ Q.T
            
            if self.fan_in_fan_out:
                # 物理存储: [in, out]
                # 计算公式: W - scale * (R.T @ Q.T)
                lhs = R_calc.T  # [in, rank]
                rhs = Q_calc.T  # [rank, out]
            else:
                # 物理存储: [out, in]
                # 计算公式: W - scale * (Q @ R)
                lhs = Q_calc      # [out, rank]
                rhs = R_calc      # [rank, in]
            
            # 获取物理存储的权重数据
            target_weight_data = self.base_layer.weight.data
            num_rows = target_weight_data.shape[0]
            chunk_size = 512 # 可以根据显存大小调整，512/1024 通常比较安全
            
            for i in range(0, num_rows, chunk_size):
                end = min(i + chunk_size, num_rows)
                
                # 计算当前 chunk 的修正值
                # [chunk_size, rank] @ [rank, cols] -> [chunk_size, cols]
                correction_chunk = (lhs[i:end] @ rhs) * scale
                
                # 原位更新，注意转换回原始 dtype (如 bf16/fp16)
                target_weight_data[i:end].sub_(correction_chunk.to(target_weight_data.dtype))
                
            # 清理
            del Q_calc, R_calc, lhs, rhs, Q_final, R_final
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"Error during optimized initialization: {e}")
            print("Falling back to random initialization.")
            # Fallback 逻辑
            self.frozen_Q[adapter_name] = torch.randn(
                self.out_features, rank, device=device, dtype=dtype
            ).detach()
            self.lora_B[adapter_name] = nn.Parameter(
                self.frozen_Q[adapter_name].clone(), requires_grad=False
            )
            self.lora_base[adapter_name] = nn.Parameter(
                torch.randn(rank, self.in_features, device=device, dtype=dtype),
                requires_grad=False
            )
            self.lora_A[adapter_name] = nn.Parameter(
                torch.zeros(rank, self.in_features, device=device, dtype=dtype)
            )
    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        
        self.enforce_triangular()
        
        base_out = self.base_layer(x)
        
        self.triu_mask = self.triu_mask.to(x.device)
        
        # Q @ (base_R + delta_R) @ x
        delta_out = (x @ (self.lora_base["default"].to(dtype) + 
                          (self.lora_A["default"].to(dtype) * self.triu_mask.to(dtype))).t() 
                     @ self.frozen_Q["default"].to(dtype).t()) * self.scaling["default"]
        
        return base_out + delta_out

def inject_delta_r_triu_lora_layer(
    model: nn.Module, 
    target_modules: Union[list[str], str],
    rank: int = 64,
    alpha: int = 64,
    delta_scale: float = 0.01,
    init_method: str = "triu_deltaR",
    use_zero_init: bool = True,
    **kwargs
):
    if isinstance(target_modules, str):
        target_modules = [target_modules]
        
    for name, module in model.named_modules():
        if any(target_key in name for target_key in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                new_layer = DeltaRTriuLoraLayer(
                    module,
                    rank=rank,
                    alpha=alpha,
                    delta_scale=delta_scale,
                    init_method=init_method,
                    use_zero_init=use_zero_init,
                    **kwargs
                )
                
                try:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, new_layer)
                except ValueError:
                    setattr(model, name, new_layer)
            
    return model