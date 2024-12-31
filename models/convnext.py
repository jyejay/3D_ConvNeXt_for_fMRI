import torch
import torch.nn as nn
from .components import LayerNorm3D, Block3D

"""
This code is modified from ConvNeXt (https://github.com/facebookresearch/ConvNeXt)
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under Apache License 2.0

Modified to support 3D operations by jyejay

"""

class ConvNeXt3D(nn.Module):
    def __init__(self, 
                 in_chans=1, 
                 num_classes=5, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6):
        super().__init__()
        
        # Downsample layers
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm3D(dims[0])
            )
        ])
        
        for i in range(1, len(dims)):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv3d(dims[i - 1], dims[i], kernel_size=2, stride=2),
                LayerNorm3D(dims[i])
            ))
            
        # Main stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(*[
                Block3D(dim=dims[i], 
                       drop_path=dp_rates[cur + j],
                       layer_scale_init_value=layer_scale_init_value) 
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
        # Final norm and head
        self.norm = LayerNorm3D(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
        x = self.norm(x)
        return x.mean([2, 3, 4])  # Global average pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x