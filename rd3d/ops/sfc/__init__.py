import torch
from typing import List
from . import sfc_cuda

"""
references: 
        http://pdebuyl.be/blog/2015/hilbert-curve.html
        https://lutanho.net/pic2html/draw_sfc.html
"""


def min_required_order(coord3d):
    # +1 to ensure 2**order > coord3d.max() is hold.
    return torch.ceil(torch.log2(coord3d[:, -3:].max() + 1)).long().item()


class HilbertEncode(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, xyz: torch.Tensor, order: int):
        # order: 0 <= max_coord < 2**order
        # TODO: merge bs_id into code and support float-type coords
        assert 1 <= order <= 20
        assert xyz.shape[-1] in [3, 4]
        assert xyz.dtype == torch.int32
        assert xyz.is_contiguous()
        codes = xyz.new_empty((xyz.shape[0]), dtype=torch.int64)
        sfc_cuda.hilbert_curve_encoder(xyz, codes, order)
        return codes


class HilbertDecode(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, codes: torch.Tensor, order: int):
        assert 1 <= order <= 20
        assert codes.dtype == torch.int64
        assert codes.is_contiguous()
        xyz = codes.new_empty((codes.shape[0], 3), dtype=torch.int32)
        sfc_cuda.hilbert_curve_decoder(codes, xyz, order)
        return xyz


class ArgSortHilbert(torch.autograd.Function):

    def forward(ctx, xyz: torch.Tensor, perm: torch.Tensor, key: torch.Tensor, key_out: torch.Tensor,
                bs_num_cum: torch.Tensor, order: int):
        # order: 0 <= max_coord < 2**order
        # TODO: merge bs_id into code and support float-type coords
        assert 1 <= order <= 20
        assert xyz.shape[-1] in [3, 4]
        assert xyz.dtype == torch.int32
        assert xyz.is_contiguous()
        assert bs_num_cum.is_contiguous()

        val_out = xyz.new_zeros((xyz.shape[0]), dtype=torch.int64)
        sfc_cuda.argsort_hilbert(xyz, key, perm, key_out, val_out, bs_num_cum, order)
        return val_out


class IndicesGrouping(torch.autograd.Function):

    def forward(ctx,
                indices: torch.Tensor,
                batch_count: torch.Tensor,
                group_size: int):
        assert indices.is_contiguous() and indices.dim() == 2 and indices.size(0) < indices.size(1)
        batch_end = torch.cumsum(batch_count, dim=0, dtype=torch.int32)
        padded_batch_count = torch.div(batch_count + group_size - 1,
                                       group_size, rounding_mode='floor') * group_size
        padded_batch_end = torch.cumsum(padded_batch_count, dim=0, dtype=torch.int32)
        padded_indices = indices.new_empty((2 * indices.size(0), padded_batch_end[-1]))
        sfc_cuda.indices_grouping_batch(batch_end, indices, padded_batch_end, padded_indices)
        return padded_indices, padded_batch_count


hilbert_curve_encoder = HilbertEncode.apply
hilbert_curve_decoder = HilbertDecode.apply
hilbert_argsort = ArgSortHilbert.apply
indices_grouping = IndicesGrouping.apply
