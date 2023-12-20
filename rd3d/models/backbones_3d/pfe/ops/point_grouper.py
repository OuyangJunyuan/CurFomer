import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .....utils.common_utils import gather, apply1d
from .....core.base import Register
from .point_querier import QUERIES

GROUPERS = Register("groupers")


class PointGrouping(nn.Module):
    def __init__(self, grouping_cfg, input_channels):
        super(PointGrouping, self).__init__()
        self.cfg = grouping_cfg
        self.need_query_features = False
        self.use_xyz = self.cfg.get('xyz', True)

        self.output_channels = self.cfg.mlps[-1]
        self.input_channels = input_channels + 3 if self.use_xyz else input_channels
        self.mlps = self.build_mlps(self.cfg.mlps, in_channels=self.input_channels)

    @staticmethod
    def build_mlps(mlp_cfg, in_channels):
        shared_mlp = []
        for k in range(len(mlp_cfg)):
            shared_mlp.extend([
                nn.Linear(in_channels, mlp_cfg[k], bias=False),
                nn.BatchNorm1d(mlp_cfg[k]),
                nn.ReLU()
            ])
            in_channels = mlp_cfg[k]
        return nn.Sequential(*shared_mlp)


@GROUPERS.register('all')
class AllPointGrouping(PointGrouping):
    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(AllPointGrouping, self).__init__(grouping_cfg, input_channels)

    def group(self, xyz: torch.Tensor, features: torch.Tensor = None, *args, **kwargs):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param features: (B, N, C) descriptors of the features
        :return:
            new_features: (B, 1, N, C{ + 3})
        """
        grouped_xyz = xyz.unsqueeze(1)  # (B,1,N,3)
        if features is not None:
            grouped_features = features.unsqueeze(1)  # (B,1,N,C)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=-1)  # (B,1,N,3+C)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features,

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None, *args, **kwargs):
        """
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param new_xyz: not used
             :param features: (B, N, C) descriptors of the features
             :return:
                 new_features: (B,  3 + C, npoint, nsample)
        """
        new_features = self.group(xyz, features)  # (B,1,N,C1)
        new_features = apply1d(self.mlps, new_features)  # (B,1,N,C2)
        new_features = torch.max(new_features, dim=2)[0]  # (B,1,C2)
        return new_features


@GROUPERS.register('ball')
class BallQueryPointGrouping(PointGrouping):
    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super().__init__(grouping_cfg, input_channels)
        self.sampler_return = {}
        self.radius = self.cfg.query.radius
        self.neighbour = self.cfg.query.neighbour

        if isinstance(self.radius, float):
            if grouping_cfg.get("use_hash", False):
                from .point_sampler import SAMPLERS
                for s in kwargs["parent"].samplers:
                    if isinstance(s, SAMPLERS["havs"]):
                        self.sampler_return = s.return_dict
                self.querier = QUERIES.build_from_name("grid_ball")
            else:
                self.querier = QUERIES.build_from_name("ball")
        elif isinstance(self.radius, list):
            self.querier = QUERIES.build_from_name("ball_dilated")
        else:
            raise NotImplemented

    def query_and_group(self, new_xyz: torch.Tensor, xyz: torch.Tensor, feats: torch.Tensor = None):
        """
         :param new_xyz:    (B, M, 3) centroids
         :param xyz:        (B, N, 3) xyz coordinates of the features
         :param feats:      (B, N, C) descriptors of the features
         :return:
             empty_mask:    (B, M) tensor with the number of grouped points for each ball query
             new_feats:     (B, M, K, {3 +} C)
        """
        group_member_cnt, group_member_ind = self.querier(
            self.radius, self.neighbour, xyz, new_xyz, **self.sampler_return
        )  # (B,M) (B,M,K)
        empty_mask = group_member_cnt > 0

        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)

        if feats is not None:
            feats = gather(feats, group_member_ind)
            new_feats = torch.cat([grouped_xyz, feats], dim=-1) if self.use_xyz else feats  # (B,M,K,{3+}C)
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_feats = grouped_xyz

        return empty_mask, new_feats

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        empty_mask, new_feats = self.query_and_group(new_xyz, xyz, feats)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None, None]  # (B,M,K,C1)
        new_feats = torch.max(new_feats, dim=2)[0]  # (B,M,Co)
        return new_feats,



@GROUPERS.register('fast-ball')
class FastBallQueryPointGrouping(PointGrouping):
    """
    accelerate mlp for efficient inference.
    Reference Paper: https://ojs.aaai.org/index.php/AAAI/article/view/16207
    Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection
    """

    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(FastBallQueryPointGrouping, self).__init__(grouping_cfg, input_channels)

        radius = self.cfg.query.radius
        self.radius = [radius] if isinstance(radius, float) else radius
        self.neighbour = self.cfg.query.neighbour
        querier_type = 'ball_dilated' if self.radius.__len__() > 1 else 'ball'
        self.querier = partial(QUERIES.build_from_name(querier_type), *self.radius, self.neighbour)

    def build_mlps(self, mlp_cfg, in_channels):
        build_mlps_func = super(FastBallQueryPointGrouping, self).build_mlps
        self.add_module('xyz_encoder', build_mlps_func(mlp_cfg[:1], 3)[:2])
        self.add_module('feat_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        self.add_module('nonlinear', nn.ReLU())
        refine_encoder = build_mlps_func(mlp_cfg[1:], mlp_cfg[0])
        return refine_encoder

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        feat_feats = apply1d(self.feat_encoder, feats)  # (B,M,C1)

        group_member_cnt, group_member_ind = self.querier(xyz, new_xyz)  # (B,M) (B,M,K)
        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)

        empty_mask = group_member_cnt > 0
        grouped_feat_feats = gather(feat_feats, group_member_ind)  # (B,M,K,C1)
        grouped_xyz_feats = apply1d(self.xyz_encoder, grouped_xyz)  # (B,M,K,C1)
        grouped_feats = self.nonlinear(grouped_xyz_feats + grouped_feat_feats)  # (B,M,K,C1)

        new_feats = torch.max(grouped_feats, dim=2)[0]  # (B,M,C1)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None]  # (B,M,C2)
        return new_feats,


@GROUPERS.register('norm-fast-ball')
class NormalizedFastBallQueryPointGrouping(PointGrouping):
    """
    add GEOMETRIC AFFINE MODULE (layer-norm in point cloud) into ball group.
    Reference Paper: https://arxiv.org/abs/1907.03670
    RETHINKING NETWORK DESIGN AND LOCAL GEOMETRY IN POINT CLOUD: A SIMPLE RESIDUAL MLP FRAMEWORK.

    Warnings: seem to have a bad performance.
    """

    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(NormalizedFastBallQueryPointGrouping, self).__init__(grouping_cfg, input_channels)

        radius = self.cfg.query.radius
        self.radius = [radius] if isinstance(radius, float) else radius
        self.neighbour = self.cfg.query.neighbour
        querier_type = 'ball_dilated' if self.radius.__len__() > 1 else 'ball'
        self.querier = partial(QUERIES.build_from_name(querier_type), *self.radius, self.neighbour)

        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, self.cfg.mlps[0]]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, self.cfg.mlps[0]]))

    def build_mlps(self, mlp_cfg, in_channels):
        build_mlps_func = super(NormalizedFastBallQueryPointGrouping, self).build_mlps
        self.add_module('xyz_encoder', build_mlps_func(mlp_cfg[:1], 3)[:2])
        self.add_module('feat_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        self.add_module('nonlinear', nn.ReLU())
        refine_encoder = build_mlps_func(mlp_cfg[1:], mlp_cfg[0])
        return refine_encoder

    def normalized(self, feats):
        center = feats[:, :, 0:1, :]
        relative = feats - center
        std = relative.view(feats.shape[0], -1).std(dim=-1)[..., None, None, None]
        feats = relative / (std + 1e-5)
        feats = self.affine_alpha * feats + self.affine_beta
        return feats

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        feat_feats = apply1d(self.feat_encoder, feats)  # (B,M,C1)

        group_member_cnt, group_member_ind = self.querier(xyz, new_xyz)  # (B,M) (B,M,K)
        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)

        empty_mask = group_member_cnt > 0
        grouped_feat_feats = gather(feat_feats, group_member_ind)  # (B,M,K,C1)
        grouped_xyz_feats = apply1d(self.xyz_encoder, grouped_xyz)  # (B,M,K,C1)
        grouped_feats = self.nonlinear(grouped_xyz_feats + grouped_feat_feats)  # (B,M,K,C1)
        grouped_feats = self.normalized(grouped_feats)

        new_feats = torch.max(grouped_feats, dim=2)[0]  # (B,M,C1)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None]  # (B,M,C2)
        return new_feats,


@GROUPERS.register('point-transformer')
class PointTransformer(PointGrouping):
    """
    local-attention point transformer
    Reference Paper: https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html
    Point Transformer
    """

    def __init__(self, grouping_cfg, input_channels, *args, **kwargs):
        super(PointTransformer, self).__init__(grouping_cfg, input_channels)

        radius = self.cfg.query.radius
        self.radius = [radius] if isinstance(radius, float) else radius
        self.neighbour = self.cfg.query.neighbour
        querier_type = 'ball_dilated' if self.radius.__len__() > 1 else 'ball'
        self.querier = partial(QUERIES.build_from_name(querier_type), *self.radius, self.neighbour)

    def build_mlps(self, mlp_cfg, in_channels):
        build_mlps_func = super(PointTransformer, self).build_mlps
        self.add_module('xyz_encoder', build_mlps_func(mlp_cfg[:1], 3)[:2])
        self.add_module('feat_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        self.add_module('attention_encoder', build_mlps_func(mlp_cfg[:1], in_channels - 3)[:2])
        refine_encoder = build_mlps_func(mlp_cfg[1:], mlp_cfg[0])
        return refine_encoder

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, feats: torch.Tensor = None, *args, **kwargs):
        """
             :param new_xyz: (B, M, 3) centroids
             :param xyz: (B, N, 3) xyz coordinates of the features
             :param feats: (B, N, Ci) descriptors of the features
             :return:
                 new_feats: (B, M, Co)
        """
        feat_feats = apply1d(self.feat_encoder, feats)  # (B,M,C1)
        attention = apply1d(self.attention_encoder, feats)  # (B,M,C1)

        group_member_cnt, group_member_ind = self.querier(xyz, new_xyz)  # (B,M) (B,M,K)
        empty_mask = group_member_cnt > 0

        grouped_xyz = gather(xyz, group_member_ind) - new_xyz.unsqueeze(-2)  # (B,M,K,3)
        grouped_feat_feats = gather(feat_feats, group_member_ind)  # (B,M,K,C1)
        grouped_attention = gather(attention, group_member_ind)  # (B,M,K,C1)

        grouped_xyz_feats = apply1d(self.xyz_encoder, grouped_xyz)  # (B,M,K,C1)
        grouped_feat_feats += grouped_xyz_feats  # (B,M,K,C1)
        grouped_attention += grouped_xyz_feats  # (B,M,K,C1)

        new_feats = (grouped_feat_feats.softmax(dim=2) * grouped_attention).sum(dim=2)  # (B,M,C1)
        new_feats = apply1d(self.mlps, new_feats) * empty_mask[..., None]  # (B,M,C2)
        return new_feats,

