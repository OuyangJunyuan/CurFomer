import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from ...utils.common_utils import gather, apply1d
from ..neck import NECKS


class Voxel2Point(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, batch_dict):


        pass
