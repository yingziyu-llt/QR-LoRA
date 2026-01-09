import torch
import torch.nn as nn
from typing import Union

class FastICA_GPU:
    def __init__(self, n_components=None, max_iter=200, tol=1e-4, device='cuda'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def _sym_decorrelation(self, W):
        # Symmetric decorrelation: W <- (W * W.T)^(-0.5) * W
        # W shape: (n_comp, n_comp)
        s, u = torch.linalg.eigh(torch.mm(W, W.t()))
        s = torch.clamp(s, min=1e-10)
        w_decorr = torch.mm(torch.mm(u, torch.diag(1.0 / torch.sqrt(s))), u.t())
        return torch.mm(w_decorr, W)

    def fit_transform_with_sorting(self, X, final_rank):
        """
        执行 ICA 并根据能量排序返回前 K 个分量
        X shape: (n_samples, n_features)
        """
        X = X.to(self.device).float()
        n_samples, n_features = X.shape
        
        # --- 优化点 1: SVD 预降维 (PCA Whitening) ---
        # 直接在 SVD 阶段截断到 n_components，大大减小后续 ICA 迭代的矩阵规模
        # 居中
        mean = torch.mean(X, dim=0, keepdim=True)
        X_centered = X - mean
        
        # 如果 n_components 未指定，或者大于特征数，则限制
        n_comp = self.n_components if self.n_components is not None else min(n_samples, n_features)
        
        # SVD: X ~ U @ S @ V.T
        # 我们只需要 V (特征方向) 和 S (能量)
        # torch.linalg.svd 在 GPU 上非常快
        try:
            u, s, vh = torch.linalg.svd(X_centered, full_matrices=False)
        except RuntimeError: 
            # 极少数情况 SVD 不收敛，回退到随机
            return None, None
            
        # 截断 SVD 到 n_comp (比如 4*rank)
        u = u[:, :n_comp]
        s = s[:n_comp]
        vh = vh[:n_comp, :] # (n_comp, n_features)
        limit = s[0] * 1e-4
        s = torch.clamp(s, min=limit)
        
        # 白化后的数据: X_white = sqrt(n) * U
        # 这样协方差矩阵为 Identity
        X1 = u.t() * (n_samples ** 0.5) # Shape: (n_comp, n_samples)
        
        # --- 优化点 2: FastICA 迭代 (在小矩阵上进行) ---
        # W shape: (n_comp, n_comp) -> 维度很小，计算极快
        W = torch.randn(n_comp, n_comp, device=self.device)
        W = self._sym_decorrelation(W)
        
        for i in range(self.max_iter):
            W_old = W.clone()
            
            # g(u) = tanh(u)
            wx = torch.mm(W, X1)
            g_wx = torch.tanh(wx)
            g_prime_wx = 1 - g_wx ** 2
            
            # Update rule
            term1 = torch.mm(g_wx, X1.t()) / n_samples
            term2 = torch.mean(g_prime_wx, dim=1, keepdim=True) * W
            W = term1 - term2
            
            W = self._sym_decorrelation(W)
            
            lim = torch.max(torch.abs(torch.abs(torch.diag(torch.mm(W, W_old.t()))) - 1))
            if lim < self.tol:
                break
        
        # --- 优化点 3: 能量排序与恢复 ---
        
        # 计算源信号 S_extracted = W @ X_white = W @ (sqrt(n)*U.T)
        # 对应的混合矩阵 A_extracted.T = pinv(W) @ S_svd @ Vh
        # 因为做了白化，实际 Mixing Matrix (A) = (W @ K)^-1 = K^-1 @ W^-1 = V @ S/sqrt(n) @ W^T
        # 简而言之，权重重构为: W_orig ~ S_sources @ A_mixing.T
        
        # 计算 Sources (n_samples, n_comp)
        S_matrix = torch.mm(W, X1).t()
        
        # 计算 Mixing Matrix A (n_features, n_comp)
        # A = V * S * W^T / sqrt(n)
        # 这里的数学推导：X ~ S @ A.T. 
        # 我们有 S = X_white.T @ W.T
        # 反解得到 A 矩阵的估计
        # 由于 W 是正交的 (decorrelated), inv(W) = W.T
        # K_inv = V @ (S / sqrt(n))
        # A = K_inv @ W.T
        
        S_diag = torch.diag(s)
        A_matrix = torch.mm(torch.mm(vh.t(), S_diag / (n_samples ** 0.5)), W.t())
        
        with torch.no_grad():
            s_norm = torch.norm(S_matrix, dim=0, keepdim=True) + 1e-6
            a_norm = torch.norm(A_matrix, dim=0, keepdim=True) + 1e-6
            
            # 计算平衡因子，使得 S 和 A 的 scale 接近
            # new_s = s / sqrt(s_norm/a_norm)
            # new_a = a * sqrt(s_norm/a_norm)
            scale_factor = torch.sqrt(s_norm / a_norm)
            
            S_matrix = S_matrix / scale_factor
            A_matrix = A_matrix * scale_factor

        # 根据 A 的列范数（能量）排序
        # A_matrix shape: (in_features, n_comp)
        energies = torch.norm(A_matrix, dim=0) # (n_comp,)
        
        sorted_indices = torch.argsort(energies, descending=True)
        top_indices = sorted_indices[:final_rank]
        
        # 截取前 final_rank 个
        S_final = S_matrix[:, top_indices] # (n_samples, rank)
        A_final = A_matrix[:, top_indices] # (in_features, rank)
        
        return S_final, A_final

class ICALoraLayer(nn.Module):
    def __init__(
        self,
        base_layer,
        rank: int = 64,
        alpha: int = 64,
        delta_scale: float = 0.01,
        init_method: str = "ica",
        use_zero_init: bool = True,
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
        
        self.lora_A = nn.ParameterDict({}) 
        self.lora_B = nn.ParameterDict({}) 
        self.lora_base = nn.ParameterDict({}) 
        self.frozen_S = {} 
        
        # 实例化一个 GPU Solver，避免重复创建，但在 Layer 里只存配置
        # 实际计算时动态创建以节省显存或重用
        self._ica_config = {
            'max_iter': 100, # 减少迭代次数，通常 GPU 上收敛很快
            'tol': 1e-3
        }

        self.init_weights("default", delta_scale)

    def init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        if self.init_method == "ica":
            self.ica_init_weights(adapter_name, delta_scale)
        else:
            raise NotImplementedError(f"Init method {self.init_method} not implemented")
        
    def ica_init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        
        base_weights = self.base_layer.weight.data
        rank = self.r[adapter_name]
        device = base_weights.device
        
        # 处理 Conv2d 或 Fan_in_fan_out
        if base_weights.dim() == 4:
            # Conv2d: (out, in, k, k) -> (out, in*k*k)
            out_ch, in_ch, k1, k2 = base_weights.shape
            W_flat = base_weights.view(out_ch, -1)
        elif self.fan_in_fan_out:
            W_flat = base_weights.T
        else:
            W_flat = base_weights

        # 确保是 float32 (ICA 对精度敏感，但 fp16 容易溢出)
        W_input = W_flat.float() 
        
        # ICA 预备维度: 稍微多取一些，比如 2 倍 rank，不用取全部维度
        # 这大大减少了计算量
        n_components_temp = min(W_input.shape[1], W_input.shape[0], 16 * rank) 
        
        # print(f"GPU ICA on {W_input.shape}, reducing to {n_components_temp} then top {rank}...")

        solver = FastICA_GPU(
            n_components=n_components_temp, 
            device=device, 
            **self._ica_config
        )

        S_final, A_final = solver.fit_transform_with_sorting(W_input, rank)
        
        if S_final is not None:
            # S_final: (out_features, rank) -> 对应 LoRA B (Output side)
            # A_final: (in_features, rank)  -> 对应 LoRA A/Base (Input side)

            # 1. 冻结部分 (Sources)
            self.frozen_S[adapter_name] = S_final.detach()
            self.lora_B[adapter_name] = nn.Parameter(S_final.detach(), requires_grad=False)
            
            # 2. 冻结部分 (Base Mixing) -> 转置为 (rank, in)
            self.lora_base[adapter_name] = nn.Parameter(A_final.t().detach(), requires_grad=False)
            
            # 3. 可训练部分 (Delta Mixing)
            if self.use_zero_init:
                self.lora_A[adapter_name] = nn.Parameter(
                    torch.zeros(rank, A_final.shape[0], device=device)
                )
            else:
                self.lora_A[adapter_name] = nn.Parameter(
                    torch.randn(rank, A_final.shape[0], device=device) * delta_scale
                )
            
            # 4. 减去初始近似值，使初始 Forward == Pretrained Weight
            # Reconstruct = S @ A.T
            # 注意 scaling
            reconstructed = torch.mm(S_final, A_final.t())
            weight_adjustment = reconstructed * self.scaling[adapter_name]
            
            # Reshape back if Conv2d
            if base_weights.dim() == 4:
                weight_adjustment = weight_adjustment.view_as(base_weights)
            elif self.fan_in_fan_out:
                weight_adjustment = weight_adjustment.T

            # In-place update base weights
            with torch.no_grad():
                self.base_layer.weight.data -= weight_adjustment.to(base_weights.dtype)
                
            # 清理显存
            del S_final, A_final, reconstructed, weight_adjustment
            
        else:
            # Fallback
            print(f"ICA failed/diverged. Using random init.")
            self.frozen_S[adapter_name] = torch.randn(self.out_features, rank, device=device)
            self.lora_B[adapter_name] = nn.Parameter(self.frozen_S[adapter_name].clone(), requires_grad=False)
            self.lora_base[adapter_name] = nn.Parameter(torch.randn(rank, self.in_features, device=device), requires_grad=False)
            self.lora_A[adapter_name] = nn.Parameter(torch.zeros(rank, self.in_features, device=device))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        # base_layer 已经被修改过了，所以 base_out 包含了 "W_orig - W_ica_init"
        base_out = self.base_layer(x)
        
        # Delta 计算
        # 公式: x @ (A_base + A_delta).T @ S.T * scale
        # 对应: x @ (lora_base + lora_A).T @ frozen_S.T
        
        A_combined = self.lora_base["default"] + self.lora_A["default"]
        S_frozen = self.frozen_S["default"]
        
        # 优化矩阵乘法顺序以减少显存
        # 先算 x @ A_combined.T -> (batch, rank)
        temp = torch.nn.functional.linear(x.to(A_combined.dtype), A_combined)
        
        # 再算 temp @ S_frozen.T -> (batch, out)
        delta_out = torch.nn.functional.linear(temp, S_frozen.to(A_combined.dtype))
        
        return base_out + delta_out * self.scaling["default"]

def inject_ica_lora_layer(
    model: nn.Module, 
    target_modules: Union[list[str], str],
    rank: int = 64,
    alpha: int = 64,
    delta_scale: float = 0.01,
    init_method: str = "ica",
    use_zero_init: bool = True,
    **kwargs
):
    if isinstance(target_modules, str):
        target_modules = [target_modules]
        
    # 收集需要修改的层，避免在遍历中修改字典导致的问题
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if any(target_key in name for target_key in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_replace.append((name, module))
    
    print(f"Injecting ICA-LoRA to {len(modules_to_replace)} layers (GPU accelerated)...")
    
    for name, module in modules_to_replace:
        new_layer = ICALoraLayer(
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
