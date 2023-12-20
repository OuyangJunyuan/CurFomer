import time

import torch
from ..model_utils import build_mlps
from ...ops import sfc
from ...utils.spconv_utils import replace_feature, spconv
from ...core.base import Register

VFL = Register('VoxFormerLayer')


@VFL.register('ResPointMLP')
class ResidualPointMLP(torch.nn.Module):
    def __init__(self, model_cfg, input_channel, group_size):
        super().__init__()
        self.norm = self.build_group_norm(input_channel)
        self.pre_blocks = self.build_res_mlps(self.norm.output_channel, model_cfg.pre)
        self.pos_blocks = self.build_res_mlps(2 * self.pre_blocks[-1].output_channel, model_cfg.pos)
        self.agg = lambda x: x.max(dim=1, keepdim=True)[0]

        self.group_size = group_size
        self.output_channel = model_cfg.pos[-1]

    @staticmethod
    def build_group_norm(input_channel):
        class GeoAffine(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output_channel = 3 + input_channel
                self.alpha = torch.nn.Parameter(torch.ones([1, 1, self.output_channel]))
                self.beta = torch.nn.Parameter(torch.zeros([1, 1, self.output_channel]))

            def forward(self, p, x):  # (m,g,c)
                x = torch.cat((p, x), dim=-1)
                std, mean = torch.std_mean(x, dim=1, keepdim=True)
                x = (x - mean) / (std + 1e-5)
                x = self.alpha * x + self.beta
                return x

        return GeoAffine()

    @staticmethod
    def build_res_mlp(input_channel, output_channel):
        class ResMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.act = torch.nn.ReLU()
                self.net1 = torch.nn.Sequential(
                    torch.nn.Linear(input_channel, output_channel, bias=False),
                    torch.nn.BatchNorm1d(output_channel, eps=1e-3, momentum=0.01),
                    self.act,
                )
                self.net2 = torch.nn.Sequential(
                    torch.nn.Linear(output_channel, output_channel, bias=False),
                    torch.nn.BatchNorm1d(output_channel, eps=1e-3, momentum=0.01)
                )

                self.identify = torch.nn.Sequential(
                    torch.nn.Linear(input_channel, output_channel, bias=False),
                    torch.nn.BatchNorm1d(output_channel, eps=1e-3, momentum=0.01)
                ) if input_channel != output_channel else torch.nn.Identity()
                self.output_channel = output_channel

            def forward(self, x):  # x(b,c)
                x = self.act(self.net2(self.net1(x)) + self.identify(x))
                return x

        return ResMLP()

    @staticmethod
    def build_res_mlps(input_channel, mlps_cfg):
        mlps = [input_channel] + mlps_cfg
        return torch.nn.Sequential(*[
            ResidualPointMLP.build_res_mlp(mlps[i - 1], mlps[i])
            for i in range(1, len(mlps))
        ])

    def group(self, x):
        return x.view(-1, self.group_size, x.size(-1))

    def flatten(self, x):
        return x.view(-1, x.size(-1))

    def forward(self, p, x):  # (n*g,3) (n*g,c)
        x = self.flatten(self.norm(self.group(p), self.group(x)))
        x = self.group(self.pre_blocks(x))
        agg = self.agg(x)
        x = torch.cat((x, agg.expand(-1, x.size(1), -1)), dim=-1)
        x = self.pos_blocks(self.flatten(x))
        return x


@VFL.register('PointMLP')
class PointNet(torch.nn.Module):
    def __init__(self, model_cfg, input_channel, group_size):
        super().__init__()
        self.norm = self.build_pos_embed(input_channel)
        self.pre_blocks = self.build_mlps(self.norm.output_channel, model_cfg.pre)
        self.pos_blocks = self.build_mlps(2 * model_cfg.pre[-1], model_cfg.pos)
        self.agg = lambda x: x.max(dim=1, keepdim=True)[0]

        self.group_size = group_size
        self.output_channel = model_cfg.pos[-1]

    @staticmethod
    def build_pos_embed(input_channel):
        class AddXYZ(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output_channel = 3 + input_channel

            def forward(self, p, x):  # (m,g,c)
                x = torch.cat((p - p.mean(dim=1, keepdim=True), x), dim=-1)
                return x

        return AddXYZ()

    @staticmethod
    def build_mlp(input_channel, output_channel):
        return torch.nn.Sequential(
            torch.nn.Linear(input_channel, output_channel, bias=False),
            torch.nn.BatchNorm1d(output_channel),
            torch.nn.ReLU()
        )

    @staticmethod
    def build_mlps(input_channel, mlps_cfg):
        mlps_cfg = [input_channel] + mlps_cfg
        return torch.nn.Sequential(*[
            PointNet.build_mlp(mlps_cfg[i - 1], mlps_cfg[i])
            for i in range(1, len(mlps_cfg))
        ])

    def group(self, x):
        return x.view(-1, self.group_size, x.size(-1))

    def flatten(self, x):
        return x.view(-1, x.size(-1))

    def forward(self, p, x):
        x = self.flatten(self.norm(self.group(p), self.group(x)))
        x = self.group(self.pre_blocks(x))
        agg = self.agg(x)
        x = torch.cat((x, agg.expand(-1, x.size(1), -1)), dim=-1)
        x = self.pos_blocks(self.flatten(x))
        return x


@VFL.register('Attention')
class Attention():
    pass


class MultiScaleLayer(torch.nn.Module):
    def __init__(self, network, strides=(4, 16)):
        super().__init__()
        assert len(strides) > 0

        self.network = network
        self.strides = strides
        self.agg = build_mlps([self.network.output_channel],
                              (1 + len(strides)) * self.network.output_channel)
        self.group_size = self.network.group_size
        self.output_channel = self.network.output_channel

    def backup_forward(self):
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

    def forward(self, p, x):  # (n*g,3) (n*g,c) [(n*g),...,(n*g)]
        feats = [self.network(p, x)]
        for stride in self.strides:
            assert self.group_size % stride == 0
            self.network.group_size = self.group_size // stride
            feats.append(self.network(p, x))
            self.network.group_size = self.group_size

        return self.agg(torch.cat(feats, dim=-1))


def highlight_group(pts, group_id, gs):
    import torch
    from rd3d.utils import viz_utils
    from matplotlib import pyplot as plt
    if group_id:
        c = torch.zeros_like(pts).view(-1, gs, 3).cpu().float()
        gc = torch.randperm(len(group_id))[:, None].repeat(1, gs).view(-1)
        gc = torch.tensor(plt.get_cmap('tab20c')(gc / gc.max())[:, :3]).float()
        c[group_id] = gc.view(-1, gs, 3)
        viz_utils.viz_scene((pts, c.view(-1, 3)))
    else:
        viz_utils.viz_scene(pts)


class CurveBackBone(torch.nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.cfg = model_cfg
        self.group_size = self.cfg.GROUP_SIZE

        self.blocks = torch.nn.ModuleList()

        for layer_cfg in self.cfg.BLOCKS:
            layer = VFL.build_from_cfg(layer_cfg,
                                       input_channel=input_channels,
                                       group_size=self.group_size)
            if self.cfg.get('MULTI_SCALE_STRIDES', ()):
                layer = MultiScaleLayer(layer, self.cfg.MULTI_SCALE_STRIDES)
            self.blocks.append(layer)
            input_channels = layer.output_channel
        self.num_point_features = input_channels
        self.grid_size = torch.tensor(grid_size, dtype=torch.int).view(1, 3).cuda()
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

        (ind1, _, ind12, ind21), vox_numbs = self.mapping(vox_numbs, vox_coors)
        ind_list = [ind1] + [ind12, ind21] * (len(self.blocks) // 2)

        for ind, block in zip(ind_list, self.blocks):
            vox_feats = vox_feats[ind]
            pts_coors = pts_coors[ind]
            vox_coors = vox_coors[ind]
            vox_feats = block(pts_coors, vox_feats)
            # highlight_group(pts_coors.split(vox_numbs.tolist(), dim=0)[3], [0], self.group_size)

        batch_dict.update({
            'voxel_numbers': vox_numbs,
            'voxel_coords': vox_coors,
            'voxel_features': vox_feats,
            'point_coords': pts_coors,
        })
        return batch_dict
