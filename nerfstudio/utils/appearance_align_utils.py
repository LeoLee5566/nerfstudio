from nerfstudio.field_components.mlp import MLP

from typing import Optional
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor,nn
import torch.nn.functional as F

class appearance_align_net(nn.Module):
    def __init__(self, in_dim:int, num_layers:int = 3,hidden_dim:int = 64,activation:Optional[nn.Module] = nn.ReLU(), out_activation:Optional[nn.Module] = nn.Sigmoid()):
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