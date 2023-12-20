import torch
from torch import nn
from typing import List


def build_mlps(mlp_cfg, in_channels, out_channels=None):
    shared_mlp = []
    for k in range(len(mlp_cfg)):
        shared_mlp.extend([
            nn.Linear(in_channels, mlp_cfg[k], bias=False),
            nn.BatchNorm1d(mlp_cfg[k]),
            nn.ReLU()
        ])
        in_channels = mlp_cfg[k]
    if out_channels:
        shared_mlp.append(nn.Linear(in_channels, out_channels, bias=True))
    return nn.Sequential(*shared_mlp)


class StackTensor:
    def __init__(self, tensors: List[torch.Tensor]):
        nums = [t.shape[0] for t in tensors]
