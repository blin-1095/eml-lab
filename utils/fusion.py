import os
import torch
from tinyyolov2 import TinyYoloV2
from typing import Dict, Any, Union

def _fuse_conv_bn(conv_w, bn_rm, bn_rv, bn_w, bn_b, conv_b=None):
    """Mathematically fuses Conv and BatchNorm layers."""
    bn_eps = 1e-05
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    
    shape = [-1] + [1] * (len(conv_w.shape) - 1)
    fused_conv = conv_w * (bn_w / torch.sqrt(bn_rv + bn_eps)).reshape(shape)
    fused_bias = (bn_w * (conv_b - bn_rm)) / torch.sqrt(bn_rv + bn_eps) + bn_b
    
    return fused_conv, fused_bias

def fuse_model_weights(sd: dict, device: torch.device) -> dict:
    """Loads an unfused state_dict and returns a fused state_dict."""
    base_sd = sd
    fused_sd = dict(base_sd)
    
    print(f"[*] Fusing Convolutional and Batchnorm weights...")
    for i in range(1, 9):
        conv_w = base_sd[f"conv{i}.weight"]
        conv_b = base_sd.get(f"conv{i}.bias", None)
        bn_w = base_sd[f"bn{i}.weight"]
        bn_b = base_sd[f"bn{i}.bias"]
        bn_rm = base_sd[f"bn{i}.running_mean"]
        bn_rv = base_sd[f"bn{i}.running_var"]
        
        fw, fb = _fuse_conv_bn(conv_w, bn_rm, bn_rv, bn_w, bn_b, conv_b)
        fused_sd[f"conv{i}.weight"] = fw
        fused_sd[f"conv{i}.bias"] = fb

        # Remove old batchnorm keys after fusing to avoid error with nn.Identity later on
        fused_sd.pop(f"bn{i}.weight", None)
        fused_sd.pop(f"bn{i}.bias", None)
        fused_sd.pop(f"bn{i}.running_mean", None)
        fused_sd.pop(f"bn{i}.running_var", None)
        fused_sd.pop(f"bn{i}.num_batches_tracked", None)
    
    return fused_sd


def fuse_sd(state_dict: Union[str, Dict[str, Any]], device: torch.device) -> dict:
    """
    Loads an unfused state dict (from memory or disk), fuses weights, 
    and returns the state dict.
    """
    # 1. Handle both file paths and in-memory dictionaries
    if isinstance(state_dict, str):
        print(f"[*] Loading unfused weights from disk: {state_dict}")
        base_sd = torch.load(state_dict, map_location=device, weights_only=True)
    elif isinstance(state_dict, dict):
        print("[*] Loading unfused weights directly from memory")
        base_sd = state_dict
    else:
        raise TypeError(f"Expected file path (str) or state dict (dict), got {type(state_dict).__name__}")

    fused_sd = fuse_model_weights(base_sd, device)
    
    return fused_sd
