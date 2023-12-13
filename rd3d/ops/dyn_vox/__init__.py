import torch
from typing import List
from . import dyn_vox_cuda


class DynamicVoxelization(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, points: torch.Tensor, voxel_size, coord_range):
        return dyn_vox_cuda.dynamic_voxelization(points, voxel_size, coord_range, 0)


class DynamicVoxelizationBatch(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, points_list: List[torch.Tensor], voxel_size, coord_range):
        for p in points_list:
            assert p.is_contiguous()
        return dyn_vox_cuda.dynamic_voxelization_batch(points_list, voxel_size, coord_range)


dyn_vox = DynamicVoxelization.apply
dyn_vox_batch = DynamicVoxelizationBatch.apply
