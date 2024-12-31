import torch
import torch.nn as nn
from timm.models.layers import DropPath

"""
The GRN (Global Response Normalization) module is modified from ConvNeXt-V2
(https://github.com/facebookresearch/ConvNeXt-V2)
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under Apache License 2.0

Modified to support 3D operations by jyejay

"""

class GRN3D(nn.Module):
    """ GRN (Global Response Normalization) layer for 3D data """
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class LayerNorm3D(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1, 1)
        return x * weight + bias

class Block3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm3D(dim)
        self.grn = GRN3D(dim*4)
        self.pwconv1 = nn.Conv3d(dim, dim * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(dim * 4, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones((dim,)) * layer_scale_init_value, 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            gamma = self.gamma.view(1, -1, 1, 1, 1).expand_as(x)
            x = gamma * x
        return input + self.drop_path(x)