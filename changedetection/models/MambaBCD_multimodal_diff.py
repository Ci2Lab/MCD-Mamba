import torch
import torch.nn.functional as F
import sys
sys.path.append('/content/drive/MyDrive')
sys.path.append('/mnt/c/GridEyeS')
import torch
import torch.nn as nn
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from MambaCD.changedetection.models.ChangeDecoder_diff_only import ChangeDecoder #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class STMambaBCD_multimodal(nn.Module):
    def __init__(self, pretrained, in_chans_opt=13, in_chans_sar=2, opt_only=False, sar_only=False, opt_bands=13, **kwargs):
        super(STMambaBCD_multimodal, self).__init__()
        
        if opt_only and sar_only:
            raise ValueError("Cannot set both opt_only and sar_only to True")
            
        self.opt_only = opt_only
        self.sar_only = sar_only
        self.opt_bands = opt_bands
        
        # Initialize encoders based on modality
        if not self.sar_only:
            self.encoder_opt = Backbone_VSSM(
                out_indices=(0, 1, 2, 3),
                pretrained=pretrained,
                in_chans=self.opt_bands,
                **kwargs
            )
            
        if not self.opt_only:
            self.encoder_sar = Backbone_VSSM(
                out_indices=(0, 1, 2, 3),
                pretrained=pretrained,
                in_chans=2,
                **kwargs
            )
        
        # Define normalization and activation layers
        _NORMLAYERS = {
            'ln': nn.LayerNorm,
            'ln2d': LayerNorm2d,
            'bn': nn.BatchNorm2d,
        }
        
        _ACTLAYERS = {
            'silu': nn.SiLU,
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
        }
        
        norm_layer = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        
        # Remove explicit args from kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        # Initialize decoder with appropriate dimensions
        if self.opt_only:
            encoder_dims = self.encoder_opt.dims
            channel_first = self.encoder_opt.channel_first
        elif self.sar_only:
            encoder_dims = self.encoder_sar.dims
            channel_first = self.encoder_sar.channel_first
        else:
            # For multimodal, we'll use the fused dimensions
            #encoder_dims = [dim // 2 for dim in self.encoder_sar.dims]
            encoder_dims = self.encoder_sar.dims
            channel_first = self.encoder_opt.channel_first
        
        # Create fusion blocks with non-linear activations
        self.fusion_block1 = nn.Sequential(
            nn.Conv2d(encoder_dims[0] * 2, encoder_dims[0], kernel_size=1),
            nn.BatchNorm2d(encoder_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_block2 = nn.Sequential(
            nn.Conv2d(encoder_dims[1] * 2, encoder_dims[1], kernel_size=1),
            nn.BatchNorm2d(encoder_dims[1]),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_block3 = nn.Sequential(
            nn.Conv2d(encoder_dims[2] * 2, encoder_dims[2], kernel_size=1),
            nn.BatchNorm2d(encoder_dims[2]),
            nn.ReLU(inplace=True)
        )
        
        self.fusion_block4 = nn.Sequential(
            nn.Conv2d(encoder_dims[3] * 2, encoder_dims[3], kernel_size=1),
            nn.BatchNorm2d(encoder_dims[3]),
            nn.ReLU(inplace=True)
        )
            
        self.decoder = ChangeDecoder(
            encoder_dims=encoder_dims,
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
        
        self.main_clf = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    
    def process_features(self, data):
        """Process features based on current modality with non-linear fusion"""
        if self.opt_only:
            return self.encoder_opt(data)
        elif self.sar_only:
            return self.encoder_sar(data)
        else:
            # Get features from both modalities
            opt_features = self.encoder_opt(data[:, :-2])
            sar_features = self.encoder_sar(data[:, -2:])
            
            # Fuse features using fusion blocks with non-linear activations
            fused_features = [
                self.fusion_block1(torch.cat([opt_features[0], sar_features[0]], dim=1)),
                self.fusion_block2(torch.cat([opt_features[1], sar_features[1]], dim=1)),
                self.fusion_block3(torch.cat([opt_features[2], sar_features[2]], dim=1)),
                self.fusion_block4(torch.cat([opt_features[3], sar_features[3]], dim=1))
            ]
            return fused_features
    
    def forward(self, pre_data, post_data):
        # Process pre-change features
        pre_features = self.process_features(pre_data)
        
        # Process post-change features
        post_features = self.process_features(post_data)
        
        # Decoder processing
        output = self.decoder(pre_features, post_features)
        
        # Final classification and upsampling
        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        
        return output