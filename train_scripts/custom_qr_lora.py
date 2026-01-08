import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer
from peft.utils.other import transpose
from typing import Optional, Union, Literal
import math
from torch.nn import Linear

import torch
import torch.nn as nn

class CustomQRLoraLayer(nn.Module):
    def __init__(
        self,
        base_layer,
        rank: int = 64,
        alpha: int = 64,
        init_method: str = "qr",
        **kwargs
    ):
        super().__init__()
        self.base_layer = base_layer
        self.init_method = init_method
        self.rank = rank
        self.alpha = alpha
        
        self.r = {"default": rank}
        self.lora_alpha = {"default": alpha}
        self.scaling = {"default": alpha / rank}
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        
        if hasattr(base_layer, "in_features"):
            self.in_features = base_layer.in_features
        else:
            self.in_features = base_layer.in_channels
        if hasattr(base_layer, "out_features"):
            self.out_features = base_layer.out_features
        else:
            self.out_features = base_layer.out_channels
            
        self.fan_in_fan_out = getattr(base_layer, "fan_in_fan_out", False)
        
        self.init_weights()
        
    def init_weights(self, adapter_name: str = "default", 
                    base_weights: Optional[torch.Tensor] = None,
                    rank: Optional[int] = None):
        if self.init_method == "qr":
            self.qr_init_weights(adapter_name, base_weights, rank)
        else:
            if rank is None:
                rank = self.r[adapter_name]
            self.lora_A[adapter_name] = nn.Parameter(torch.randn(rank, self.in_features) * 0.02)
            self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, rank) * 0.02)
    
    def qr_init_weights(self, adapter_name: str = "default", 
                       base_weights: Optional[torch.Tensor] = None,
                       rank: Optional[int] = None):
        if base_weights is None:
            base_weights = self.base_layer.weight.data
            
        if rank is None:
            rank = self.r[adapter_name]
            
        if self.fan_in_fan_out:
            base_weights = base_weights.T
            
        try:
            original_dtype = base_weights.dtype
            base_weights = base_weights.to(torch.float32)

            V, S, Uh = torch.linalg.svd(base_weights, full_matrices=False)
            
            Vr = V[:, :rank]  # (out_features, rank)
            Sr = S[:rank]     # (rank,)
            Uhr = Uh[:rank, :]  # (rank, in_features)
            del V, S, Uh
            torch.cuda.empty_cache()
            
            Sr = Sr / self.scaling[adapter_name]
            
            temp = Vr @ torch.diag(Sr)
            core_matrix = temp @ Uhr
            del Vr, Sr, Uhr, temp 
            # torch.cuda.empty_cache()
            
            Q, R = torch.linalg.qr(core_matrix, mode='reduced')
            del core_matrix
            # torch.cuda.empty_cache()
            
            self.lora_B[adapter_name] = nn.Parameter(Q[:, :rank])
            self.lora_A[adapter_name] = nn.Parameter(R[:rank, :])
            
            assert self.lora_A[adapter_name].shape[0] == rank, \
                f"lora_A first dimension mismatch: expected {rank}, got {self.lora_A[adapter_name].shape[0]}"
            assert self.lora_B[adapter_name].shape == (self.out_features, rank), \
                f"lora_B shape mismatch: expected ({self.out_features}, {rank}), got {self.lora_B[adapter_name].shape}"
            
            reconstructed = Q[:, :rank] @ R[:rank, :]
            weight = base_weights - self.scaling[adapter_name] * reconstructed
            del Q, R, reconstructed 
            torch.cuda.empty_cache()

            weight = weight.to(original_dtype)
            if self.fan_in_fan_out:
                weight = weight.T
                
            self.base_layer.weight.data = weight
            
        except RuntimeError as e:
            print(f"Error during QR initialization: {e}")
            print(f"Weight shape: {base_weights.shape}")
            print(f"Rank: {rank}")
            print(f"in_features: {self.in_features}, out_features: {self.out_features}")
            self.lora_A[adapter_name] = nn.Parameter(torch.randn(rank, self.in_features, device=base_weights.device) * 0.02)
            self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, rank, device=base_weights.device) * 0.02)

    def forward(self, x: torch.Tensor):
        base_output = self.base_layer(x)
        
        lora_output = (x @ self.lora_A["default"].t() @ self.lora_B["default"].t()) * self.scaling["default"]
        
        return base_output + lora_output

def inject_custom_qrlora_layer(model: nn.Module, 
                           target_modules: Union[list[str], str],
                           rank: int = 64,
                           alpha: int = 64,
                           init_method: str = "qr",
                           **kwargs):
    if isinstance(target_modules, str):
        target_modules = [target_modules]
        
    for name, module in model.named_modules():
        if any(target_key in name for target_key in target_modules):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if init_method == "qr":
                    new_layer = CustomQRLoraLayer(
                        module,
                        rank=rank,
                        alpha=alpha,
                        init_method=init_method,
                        **kwargs
                    )
                else:
                    from peft import LoraConfig, get_peft_model
                    config = LoraConfig(
                        r=rank,
                        lora_alpha=alpha,
                        target_modules=[name],
                        init_lora_weights=init_method,
                        **kwargs
                    )
                    new_layer = get_peft_model(module, config)
                
                try:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, new_layer)
                except ValueError:
                    setattr(model, name, new_layer)
            
    return model 
