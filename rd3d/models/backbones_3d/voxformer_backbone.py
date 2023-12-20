import time

import torch
from ..model_utils import build_mlps
from ...ops import sfc
from ...utils.spconv_utils import replace_feature, spconv
from ...core.base import Register

class Layer(torch.nn.Module):
    def __init__(self, input_channel, output_channels):
        super().__init__()
        self.mlp1 = build_mlps([output_channels], input_channel)
        self.mlp2 = build_mlps([output_channels], output_channels)

    def forward(self, x):
        n, g, c = x.size()
        loc = self.mlp1(x.view(-1, c)).view(n, g, -1)
        n, g, c = loc.size()
        glb = loc.max(dim=1)[0][:, None, :]
        x = self.mlp2((loc + glb).view(-1, c)).view(n, g, -1)
        return x


class Block(torch.nn.Module):
    def __init__(self, cfg, input_channel, group_size):
        super().__init__()
        self.group_size = group_size
        self.premap = build_mlps(cfg[:1], input_channel)
        self.layers = torch.nn.Sequential(
            *[Layer(cfg[i], cfg[i + 1])
              for i in range(0, len(cfg) - 1)]
        )
        self.enpos = build_mlps(cfg[:1], 3)

    def group(self, x):
        return x.view(-1, self.group_size, x.shape[-1])

    def flatten(self, x):
        return x.view(-1, x.shape[-1])

    def forward(self, p, x, i_to, i_from):
        # x = self.premap(x)
        #
        # p1 = p
        # e1 = self.group(p1)
        # e1 = self.enpos(self.flatten(e1 - e1.mean(dim=1, keepdim=True)))
        #
        # p2 = p1[i_to]
        # e2 = self.group(p2)
        # e2 = self.enpos(self.flatten(e2 - e2.mean(dim=1, keepdim=True)))
        #
        # x1 = x
        # x1 = x1 + self.flatten(self.layers(self.group(e1 * x1)))  # (kg,c) -> (k,g,c)
        # x2 = x1[i_to]
        # x2 = x2 + self.flatten(self.layers(self.group(e2 * x2)))

        x = self.premap(x)

        p1 = p
        e1 = self.group(p1)
        e1 = self.enpos(self.flatten(e1 - e1.mean(dim=1, keepdim=True)))

        p2 = p1[i_to]
        e2 = self.group(p2)
        e2 = self.enpos(self.flatten(e2 - e2.mean(dim=1, keepdim=True)))

        x1 = self.flatten(self.layers(self.group(e1 * x)))  # (kg,c) -> (k,g,c)
        x2 = self.flatten(self.layers(self.group(e2 * x1[i_to])))
        return p2, x2, i_from, i_to


class CurveBackBone(torch.nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.cfg = model_cfg
        self.group_size = self.cfg.GROUP_SIZE
        self.grid_size = torch.tensor(grid_size, dtype=torch.int).view(1, 3).cuda()
        mlps = self.cfg.BLOCKS

        self.blocks = torch.nn.ModuleList()
        for mlp in mlps:
            self.blocks.append(Block(mlp, input_channels, self.group_size))
            input_channels = mlp[-1]
        self.num_point_features = input_channels
        self.order = sfc.min_required_order(self.grid_size)

    def mapping(self, vox_numbs, vox_coors):
        vox_coors1 = vox_coors  # (m,4)
        if self.cfg.MULTI_GROUP == 'reverse_coors':
            vox_coors2 = vox_coors[:, [0, 3, 2, 1]]  # (m,4)
        elif self.cfg.MULTI_GROUP == 'flip_coors':
            flip_xyz = self.grid_size - vox_coors[:, 1:]
            vox_coors2 = torch.cat((vox_coors[:, :1], flip_xyz), dim=-1)  # (m,4)
        elif self.cfg.MULTI_GROUP == 'shift':
            vox_coors2 = torch.cat((vox_coors[:, :1], vox_coors[:, 1:2], vox_coors[:, 2:] + 1), dim=-1)  # (m,4)
        else:
            raise NotImplementedError
        multi_vox_coors = torch.cat((vox_coors1, vox_coors2), dim=0)  # (2m,4)
        curve_codes = sfc.hilbert_curve_encoder(multi_vox_coors, self.order).view(2, -1)  # (2,m)
        vox_indices = torch.argsort(curve_codes, dim=-1)  # (2,m)
        indices, new_vox_numbs = sfc.indices_grouping(vox_indices, vox_numbs, self.group_size)
        return indices, new_vox_numbs  # (4,kg), (bs)

    def forward(self, batch_dict):
        vox_numbs = batch_dict['voxel_numbers']
        vox_coors = batch_dict['voxel_coords']
        vox_feats = batch_dict['voxel_features']
        pts_coors = batch_dict['point_coords']

        (ind1, ind2, ind12, ind21), vox_numbs = self.mapping(vox_numbs, vox_coors)

        pts_coors = pts_coors[ind1]
        vox_coors = vox_coors[ind1]
        vox_feats = vox_feats[ind1]

        for i, block in enumerate(self.blocks):
            pts_coors, vox_feats, ind12, ind21 = block(pts_coors, vox_feats, ind12, ind21)

        batch_dict.update({
            'voxel_numbers': vox_numbs,
            'voxel_coords': vox_coors,
            'voxel_features': vox_feats,
            'point_coords': pts_coors,
        })
        # highlight_group(pts_coors.split(vox_numbs.tolist(), dim=0)[3], [0], self.group_size)
        return batch_dict
