import os
import math
import torch
import psutil
import numpy as np
from fvcore.nn.flop_count import flop_count


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage


def get_flop_stats(model, is_3d):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        is_3d (bool): if True, prepare data for 3d convolution. Otherwise,
            prepare data for 2d convolution.

    Returns:
        float: the total number of gflops of the given model.
    """
    device = torch.cuda.current_device()

    if is_3d:
        flop_inputs = [torch.rand(3, 16, 224, 224).unsqueeze(0).to(device)]
    else:
        flop_inputs = torch.rand(3, 224, 224).unsqueeze(0).to(device)

    gflop_dict, _ = flop_count(model, (flop_inputs, ))
    gflops = sum(gflop_dict.values())
    return gflops


def log_model_info(model, is_3d=False):
    """
    Log info, includes number of parameters, gpu usage and gflops.
    Args:
        model (model): model to log the info.
        is_3d (bool): if True, prepare data for 3d convolution. Otherwise,
            prepare data for 2d convolution.
    """
    print("Params: {:,}".format(params_count(model)))
    print("GPU Mem: {:,} MB".format(gpu_mem_usage()))
    print("FLOPs: {:,} GFLOPs\n".format(get_flop_stats(model, is_3d)))
