import warnings

import torch
from .vfe_template import VFETemplate
from ....ops.dyn_vox import dyn_vox_batch


class DynamicMeanVFEV2(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def get_output_feature_dim(self):
        return self.num_point_features + 1

    @torch.no_grad()
    def forward(self, batch_dict, **kwargs):
        points_list = batch_dict['points_list']
        vc, vf, vn = dyn_vox_batch(points_list, self.voxel_size, self.point_cloud_range)
        # if (vf[:, -1].int() == 0).any():
        #     warnings.warn(f"{(vf[:, -1].int() == 0).sum()} empty voxels found", RuntimeWarning)
        batch_dict['voxel_coords'] = vc  # (n1+...+nb,4)
        batch_dict['voxel_features'] = vf  # (n1+...+nb,c+1)
        batch_dict['voxel_numbers'] = vn  # (b)
        batch_dict['point_coords'] = vf[:, :3].detach().clone()  # (n1+...+nb,3)
        return batch_dict
