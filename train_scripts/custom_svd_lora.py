import torch
import torch.nn as nn
from typing import Union

class SVDLoraLayer(nn.Module):
    def __init__(
        self,
        base_layer,
        rank: int = 64,
        alpha: int = 64,
        delta_scale: float = 0.01,
        init_method: str = "svd",
        **kwargs
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        # SVD Basis: W ~ U @ (S @ V.T)
        # We fix U (The principal directions) and train the rest
        self.frozen_U = nn.ParameterDict({})  # Fixed Basis
        self.lora_M = nn.ParameterDict({})    # Trainable Coefficients (incorporates Sigma and V.T)
        self.lora_base_M = nn.ParameterDict({}) # Fixed Base Coefficients
        
        if hasattr(base_layer, "in_features"):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        elif hasattr(base_layer, "in_channels"):
            self.in_features = base_layer.in_channels
            self.out_features = base_layer.out_channels
            
        self.fan_in_fan_out = getattr(base_layer, "fan_in_fan_out", False)
        
        self.init_weights("default", delta_scale)

    def init_weights(self, adapter_name="default", delta_scale=0.01):
        # SVD Initialization
        with torch.no_grad():
            base_weights = self.base_layer.weight.data
            device = base_weights.device
            dtype = base_weights.dtype
            
            # Handle Conv2d or Transpose
            if base_weights.dim() == 4:
                # (out, in, k, k) -> (out, -1)
                W_flat = base_weights.view(base_weights.shape[0], -1)
            elif self.fan_in_fan_out:
                W_flat = base_weights.T
            else:
                W_flat = base_weights
            
            # Perform SVD: W = U @ S @ Vh
            # float32 for precision
            U, S, Vh = torch.linalg.svd(W_flat.to(torch.float32), full_matrices=False)
            
            # Truncate to rank
            U_r = U[:, :self.rank]          # (out, rank)
            S_r = S[:self.rank]             # (rank,)
            Vh_r = Vh[:self.rank, :]        # (rank, in)
            
            # 1. Define Fixed Basis (U)
            # U captures the most important output directions
            self.frozen_U[adapter_name] = nn.Parameter(U_r.contiguous(), requires_grad=False)
            
            # 2. Define Base Coefficients (M = S @ Vh)
            # M captures the input mapping and scaling
            M_base = torch.diag(S_r) @ Vh_r # (rank, in)
            self.lora_base_M[adapter_name] = nn.Parameter(M_base.contiguous(), requires_grad=False)
            
            # 3. Define Trainable Delta M
            # We train perturbations in the coefficient space
            self.lora_M[adapter_name] = nn.Parameter(
                torch.zeros_like(M_base) if delta_scale == 0 else torch.randn_like(M_base) * delta_scale
            )
            
            # 4. Subtract initialization from base weights to ensure identity start
            # W_recon = U_r @ M_base
            W_recon = U_r @ M_base
            
            # Restore shape
            if base_weights.dim() == 4:
                W_recon = W_recon.view_as(base_weights)
            elif self.fan_in_fan_out:
                W_recon = W_recon.T
                
            self.base_layer.weight.data -= (W_recon.to(dtype) * self.scaling)
            
            del U, S, Vh, W_flat, W_recon
            torch.cuda.empty_cache()

    def forward(self, x):
        adapter_name = "default"
        base_out = self.base_layer(x)
        
        # Formula: W_adapter = U_fixed @ (M_base + Delta_M)
        # Output = x @ W_adapter.T 
        #        = x @ (M_base + Delta_M).T @ U_fixed.T
        
        U_fixed = self.frozen_U[adapter_name].to(x.dtype)
        M_total = self.lora_base_M[adapter_name].to(x.dtype) + self.lora_M[adapter_name].to(x.dtype)
        
        # 1. Project input to rank space: x @ M.T -> (batch, rank)
        rank_space = torch.nn.functional.linear(x, M_total)
        
        # 2. Project back to output space: rank_space @ U.T -> (batch, out)
        # U is (out, rank), so linear uses U (which is W.T shape)
        delta_out = torch.nn.functional.linear(rank_space, U_fixed)
        
        return base_out + delta_out * self.scaling

def inject_svd_lora_layer(model, target_modules, rank=64, alpha=64, **kwargs):
    # Standard injection logic (same as your other scripts)
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                new_layer = SVDLoraLayer(module, rank=rank, alpha=alpha, **kwargs)
                parent_name, child_name = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_name), child_name, new_layer)
    return model