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
        
        # 注册 mask 为 buffer，随模型设备移动
        self.register_buffer('triu_mask', torch.triu(torch.ones(rank, self.in_features)))
        
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({}) 
        self.lora_base = nn.ParameterDict({})
        
        # [优化 1] 删除 self.frozen_Q，只用 lora_B
        # self.frozen_Q = {} 
        
        self.init_weights("default", delta_scale)

    def init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        if self.init_method == "triu_deltaR":
            self.deltaR_init_weights(adapter_name, delta_scale)
        else:
            raise NotImplementedError(f"Init method {self.init_method} not implemented")
        
    def deltaR_init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        # 确保不需要梯度计算，节省显存
        with torch.no_grad():
            base_weights = self.base_layer.weight.data
            rank = self.r[adapter_name]
            device = base_weights.device
            
            # 确保 mask 在正确设备
            if self.triu_mask.device != device:
                self.triu_mask = self.triu_mask.to(device)
            
            if self.fan_in_fan_out:
                base_weights = base_weights.T
                
            try:
                original_dtype = base_weights.dtype
                # SVD 需要 float32 以保证精度
                base_weights_fp32 = base_weights.to(torch.float32)
                
                U, S, Vh = torch.linalg.svd(base_weights_fp32, full_matrices=False)
                
                core_U = U[:, :rank]
                core_S = S[:rank]
                core_Vh = Vh[:rank, :]
                
                # 显式删除大矩阵并清空缓存
                del U, S, Vh, base_weights_fp32
                torch.cuda.empty_cache()
                
                core_S = core_S / self.scaling[adapter_name]
                
                temp = core_U @ torch.diag(core_S)
                core_matrix = temp @ core_Vh
                del core_U, core_S, core_Vh, temp
                
                Q, R = torch.linalg.qr(core_matrix, mode='reduced')
                del core_matrix
                
                # [优化 2] 直接存入 lora_B，不再存入 frozen_Q
                # 这里的 Q 已经 detached 了，因为上面是在 no_grad 下运行的
                self.lora_B[adapter_name] = nn.Parameter(Q[:, :rank].contiguous(), requires_grad=False)
                
                self.lora_base[adapter_name] = nn.Parameter(R[:rank, :].contiguous(), requires_grad=False)
                
                if self.use_zero_init:
                    self.lora_A[adapter_name] = nn.Parameter(
                        torch.zeros(rank, self.in_features, device=device)
                    )
                else:
                    delta_R = torch.randn(rank, self.in_features, device=device) * delta_scale
                    # 初始时就应用 mask
                    self.lora_A[adapter_name] = nn.Parameter(delta_R * self.triu_mask)
                
                # 使用 lora_B 计算重建
                reconstructed = self.lora_B[adapter_name] @ self.lora_base[adapter_name]
                weight = base_weights.to(torch.float32) - self.scaling[adapter_name] * reconstructed
                
                del Q, R, reconstructed
                torch.cuda.empty_cache()
                
                weight = weight.to(original_dtype)
                if self.fan_in_fan_out:
                    weight = weight.T
                    
                self.base_layer.weight.data = weight
                
            except RuntimeError as e:
                print(f"Error during initialization: {e}")
                # Fallback logic
                fallback_Q = torch.randn(self.out_features, rank, device=device)
                self.lora_B[adapter_name] = nn.Parameter(fallback_Q, requires_grad=False)
                self.lora_base[adapter_name] = nn.Parameter(
                    torch.randn(rank, self.in_features, device=device),
                    requires_grad=False
                )
                self.lora_A[adapter_name] = nn.Parameter(
                    torch.zeros(rank, self.in_features, device=device)
                )

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        adapter_name = "default"
        
        base_out = self.base_layer(x)
        
        # [优化 3] 移除 enforce_triangular 函数调用
        # 直接在计算中应用 mask，避免修改 .data 和额外的函数开销
        
        # 确保 mask 设备正确 (通常 register_buffer 会自动处理，但为了安全)
        # mask = self.triu_mask.to(x.device) # buffer 应该已经自动跟随模型设备了，如果报错再加
        
        # A_masked = A * mask
        # 强转 dtype 以匹配输入 x (如 bf16)
        A_masked = self.lora_A[adapter_name].to(dtype) * self.triu_mask.to(dtype)
        
        # 公式: delta = (x @ (base_R + A_masked).T) @ Q.T * scale
        # 使用 lora_B 替代 frozen_Q
        
        # Step 1: Combine R
        R_combined = self.lora_base[adapter_name].to(dtype) + A_masked
        
        # Step 2: Compute
        # 注意：这里利用结合律，先算 x @ R.T (维度较小: BxLxR)，再 @ Q.T (维度变大: BxLxD)
        # 这样比先算 (R.T @ Q.T) 要省显存，前提是 R 远小于 D
        delta_out = (x @ R_combined.t()) @ self.lora_B[adapter_name].to(dtype).t()
        
        return base_out + delta_out * self.scaling[adapter_name]

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