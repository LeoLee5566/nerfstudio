from nerfstudio.field_components.mlp import MLP

from typing import Optional
import numpy as np
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from nerfstudio.utils.rich_utils import CONSOLE

class appearance_align_net(nn.Module):
    def __init__(self, in_dim:int, num_layers:int = 3,hidden_dim:int = 32,activation:Optional[nn.Module] = nn.ReLU(), out_activation:Optional[nn.Module] = nn.Sigmoid()):
        super(appearance_align_net,self).__init__()
        self.mlp = MLP(
            in_dim=in_dim,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=in_dim,
            activation=activation,
            out_activation=out_activation
        )
        
    def forward(self,appearance_code:Tensor):
        return self.mlp(appearance_code)
    
    def get_loss(self,model,camera_ray_bundle,appearance_code:Tensor,target:Tensor):
        rgb = model.get_rgb_for_appearance(camera_ray_bundle,appearance_code)
        assert rgb.shape[-1]==3 and target.shape[-1]==3,'image dimension misaligned'
        rgb = rgb.permute(2,0,1)
        target = target.permute(2,0,1)
        return F.mse_loss(rgb,target)