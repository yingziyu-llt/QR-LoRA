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
        
    def deltaR_init_weights(self, adapter_name: str = "default", delta_scale: float = 0.01):
        
        base_weights = self.base_layer.weight.data
        rank = self.r[adapter_name]
        device = base_weights.device
        
        self.triu_mask = self.triu_mask.to(device)
        
        if self.fan_in_fan_out:
            base_weights = base_weights.T
            
        try:
            original_dtype = base_weights.dtype
            base_weights = base_weights.to(torch.float32)
            
            U, S, Vh = torch.linalg.svd(base_weights, full_matrices=False)
            
            core_U = U[:, :rank].float()
            core_S = S[:rank].float()
            core_Vh = Vh[:rank, :].float()
            del U, S, Vh
            # torch.cuda.empty_cache()
            
            core_S = core_S / self.scaling[adapter_name]
            
            temp = core_U @ torch.diag(core_S)
            core_matrix = temp @ core_Vh
            del core_U, core_S, core_Vh, temp
            # torch.cuda.empty_cache()
            
            Q, R = torch.linalg.qr(core_matrix, mode='reduced')
            del core_matrix
            # torch.cuda.empty_cache()
            
            self.frozen_Q[adapter_name] = Q[:, :rank].detach()
            self.lora_B[adapter_name] = nn.Parameter(Q[:, :rank].detach(), requires_grad=False)
            
            self.lora_base[adapter_name] = nn.Parameter(R[:rank, :].detach(), requires_grad=False)
            
            if self.use_zero_init:
                self.lora_A[adapter_name] = nn.Parameter(
                    torch.zeros(rank, self.in_features, device=device)
                )
            else:
                delta_R = torch.randn(rank, self.in_features, device=device) * delta_scale
                self.lora_A[adapter_name] = nn.Parameter(delta_R * self.triu_mask)
            
            reconstructed = Q[:, :rank] @ R[:rank, :]
            weight = base_weights - self.scaling[adapter_name] * reconstructed
            del Q, R, reconstructed
            torch.cuda.empty_cache()
            
            weight = weight.to(original_dtype)
            if self.fan_in_fan_out:
                weight = weight.T
                
            self.base_layer.weight.data = weight
            
        except RuntimeError as e:
            print(f"Error during initialization: {e}")
            self.frozen_Q[adapter_name] = torch.randn(
                self.out_features, rank, device=device
            ).detach()
            self.lora_B[adapter_name] = nn.Parameter(
                self.frozen_Q[adapter_name].clone(),
                requires_grad=False
            )
            self.lora_base[adapter_name] = nn.Parameter(
                torch.randn(rank, self.in_features, device=device),
                requires_grad=False
            )
            self.lora_A[adapter_name] = nn.Parameter(
                torch.zeros(rank, self.in_features, device=device)
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