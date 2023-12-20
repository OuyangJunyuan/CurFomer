import os
import torch
from rd3d import PROJECT_ROOT
from rd3d.core.base.timer import ScopeTimer
from rd3d.ops.dyn_vox import dyn_vox, dyn_vox_batch

os.chdir(PROJECT_ROOT)
batch_dict = torch.load("data/cache/batch_dict.pth")
points = batch_dict['points']
points1 = points[points[:, 0] == 0, 1:].contiguous()
points2 = points[points[:, 0] == 1, 1:].contiguous()
points3 = points[points[:, 0] == 0, 1:].contiguous()
points4 = points[points[:, 0] == 1, 1:].contiguous()
points = points[:, 1:].clone().contiguous()
points_list = [points1, points2, points1, points2, points1, points2]

from rd3d.utils.viz_utils import viz_scene
from rd3d.ops import sfc

vox_coors, f, vox_numbs = dyn_vox_batch(points_list, [0.2, 0.2, 0.2], [-200, -200, -200, 200, 200, 200])
print(vox_numbs)
print(vox_numbs.sum())
group_size = 32
grid_size = torch.tensor([100, 100, 100]).int().cuda()


def f():
    order = sfc.min_required_order(vox_coors)
    vox_coors2 = vox_coors.detach().clone()
    vox_coors2[:, 1:] = grid_size - vox_coors2[:, 1:]
    vox_coors_numtiple = torch.cat((vox_coors, vox_coors2))
    codes = sfc.hilbert_curve_encoder(vox_coors_numtiple, order)[None, :].view(2, -1)
    indices = torch.argsort(codes, dim=-1)
    out, padded_vox_numbs = sfc.indices_grouping(indices, vox_numbs, group_size)
    ind1, ind2, ind12, ind21 = out
    begin = 0
    for bid in range(batch_dict['batch_size']):
        end = begin + padded_vox_numbs[bid]

        c1 = vox_coors[ind1][begin:end]
        assert (c1[:, 0] == bid).all()
        c2 = vox_coors[ind2][begin:end]
        assert (c2[:, 0] == bid).all()

        c1 = vox_coors[ind1]
        c2 = vox_coors[ind2]
        assert (c1[ind12][ind21] == c1).all()

        begin = end

for _ in range(1000):
    f()

for _ in range(1000):
    with ScopeTimer("", average=True, verbose=False) as t:
        f()
print(t.duration)

# order = sfc.min_required_order(vox_coors)
# vox_coors2 = vox_coors.detach().clone()
# vox_coors2[:, 1:] = grid_size - vox_coors2[:, 1:]
# vox_coors_numtiple = torch.cat((vox_coors, vox_coors2))
# codes = sfc.hilbert_curve_encoder(vox_coors_numtiple, order)[None, :].view(2, -1)
# indices = torch.argsort(codes, dim=-1)
# out, padded_vox_numbs = sfc.indices_grouping(indices, vox_numbs, group_size)
# ind1, ind2, ind12, ind21 = out


# assert (vox_coors[ind1][:, 0] == vox_coors[ind2][:, 0]).all()
