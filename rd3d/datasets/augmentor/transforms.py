import torch
import numpy as np

from scipy.spatial.transform import Rotation
from ...core.base import Register

AUGMENTOR = Register('augmentor')


def merge_dicts(list_of_dict):
    import pandas as pd

    return pd.DataFrame(list(list_of_dict)).to_dict(orient="list")


class AugUtils:

    def params(self, data_dict):
        return {}

    @staticmethod
    def uniform(a, b, size=None, **kwargs):
        return (b - a) * torch.rand(size or [], **kwargs) + a

    @staticmethod
    def normal(mean, std, **kwargs):
        return torch.normal(torch.tensor(mean), torch.tensor(std), **kwargs)

    @staticmethod
    def enable(prob, size=None, **kwargs):
        prob = torch.clip(torch.tensor(prob), 0, 1)
        mask = torch.rand(size or [], **kwargs) < prob
        return mask

    @staticmethod
    def angle_out_of_range(x, mid, width):
        diff = (x - mid + np.pi) % (2 * np.pi) - np.pi
        return torch.logical_or(diff < -width / 2, width / 2 < diff)

    @staticmethod
    def try_points_in_boxes_masks_from_cache(data_dict):
        def points_in_boxes():
            if 'points' in data_dict and 'gt_boxes' in data_dict:
                points: torch.Tensor = data_dict['points'][:, 0:3]
                boxes: torch.Tensor = data_dict['gt_boxes']
                if points.device == torch.device("cpu"):
                    from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
                    flags = points_in_boxes_cpu(points, boxes)
                    box_id, point_id = torch.split(flags.nonzero(), 1, dim=-1)
                    mask = torch.ones([points.size(0)], dtype=torch.int32) * -1
                    mask[point_id] = box_id.int()
                else:
                    from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
                    mask = points_in_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0)).squeeze(0)
                return [(mask == i).nonzero().squeeze(1) for i in range(boxes.size(0))]
            else:
                return []

        logs = data_dict['augment_logs']
        if 'points_in_boxes' not in logs:
            logs['points_in_boxes'] = points_in_boxes()
        return logs['points_in_boxes']


class Augmentor(AugUtils):
    def __init__(self, kwargs):
        self.name = type(self).__name__
        self.prob = kwargs.get('prob', 1.0)

    def build_params(self, params):
        from easydict import EasyDict
        if isinstance(params, list):
            params = merge_dicts(params)
        return EasyDict(params)

    def invert(self, data_dict, params=None):
        params = self.build_params(params or data_dict['augment_logs'][self.name])
        assert hasattr(self, 'backward')
        self.backward(data_dict, **params)
        return data_dict

    def __call__(self, data_dict, params=None):
        if 'augment_logs' not in data_dict:
            data_dict['augment_logs'] = {}
        if self.enable(self.prob):
            params = self.build_params(params or self.params(data_dict))
            self.forward(data_dict, **params)
            data_dict['augment_logs'][self.name] = params
        return data_dict


@AUGMENTOR.register('global_rotate')
class GlobalRotate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [-np.pi / 4, np.pi / 4])
        assert len(self.range) == 2

    @staticmethod
    def euler2matrix(euler, **kwargs):
        rot_mat = Rotation.from_euler('z', euler.numpy()).as_matrix()
        rot_mat = torch.tensor(rot_mat, **kwargs)
        return rot_mat

    @staticmethod
    def rotate_points(points, rot):
        coords, feats = points[:, :3], points[:, 3:]
        rot = GlobalRotate.euler2matrix(rot, dtype=points.dtype, device=points.device)
        coords = coords @ rot.transpose(0, 1)
        points = torch.cat((coords, feats), dim=-1)
        return points

    @staticmethod
    def rotate_boxes(boxes, rot, locally=False):
        from ...utils.common_utils import limit_period
        if not locally:
            boxes[:, :3] = GlobalRotate.rotate_points(boxes[:, :3], rot)
        boxes[:, 6] = limit_period(boxes[:, 6] + rot.to(boxes.device), offset=0.5, period=2 * np.pi)
        if boxes.size(-1) > 7:
            velo_xy = torch.cat([boxes[:, 7:9], torch.zeros_like(boxes[:, :1])], dim=-1)
            boxes[:, 7:9] = GlobalRotate.rotate_points(velo_xy, rot)[:, :2]
        return boxes

    def params(self, data_dict):
        return dict(rot=self.uniform(self.range[0], self.range[1]))

    def backward(self, data_dict, rot):
        self.forward(data_dict, -rot)

    def forward(self, data_dict, rot):
        if 'points' in data_dict:
            data_dict['points'] = self.rotate_points(data_dict['points'], rot)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.rotate_boxes(data_dict['gt_boxes'], rot)


@AUGMENTOR.register('global_translate')
class GlobalTranslate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.std = kwargs.get('std', [0, 0, 0])
        assert len(self.std) == 3

    @staticmethod
    def translate_boxes(boxes, trans):
        boxes[:, :3] += trans.to(boxes.device)
        return boxes

    @staticmethod
    def translate_points(points, trans):
        points[:, :3] += trans.to(points.device)
        return points

    def params(self, data_dict):
        return dict(trans_noise=self.normal(0.0, self.std))

    def backward(self, data_dict, trans_noise):
        self.forward(data_dict, -trans_noise)

    def forward(self, data_dict, trans_noise):
        if 'points' in data_dict:
            data_dict['points'] = self.translate_points(data_dict['points'], trans_noise)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.translate_boxes(data_dict['gt_boxes'], trans_noise)


@AUGMENTOR.register('global_scale')
class GlobalScale(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [0.95, 1.05])

    @staticmethod
    def scale_boxes(boxes, scale):
        scale = scale.to(boxes.device)
        boxes[:, :6] *= scale
        boxes[:, 7:] *= scale
        return boxes

    @staticmethod
    def scale_points(points, scale):
        scale = scale.to(points.device)
        points[:, :3] *= scale
        return points

    def params(self, data_dict):
        return dict(scale_noise=self.uniform(self.range[0], self.range[1]))

    def backward(self, data_dict, scale_noise):
        self.forward(data_dict, 1.0 / scale_noise)

    def forward(self, data_dict, scale_noise):
        if 'points' in data_dict:
            data_dict['points'] = self.scale_points(data_dict['points'], scale_noise)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.scale_boxes(data_dict['gt_boxes'], scale_noise)


@AUGMENTOR.register('global_flip')
class GlobalFlip(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.axis = kwargs.get('axis', ['x'])
        self.axis_prob = kwargs.get('axis_prob', 0.5)

    @staticmethod
    def flip_points(points, axis_x, axis_y):
        if axis_x:
            points[:, 1] *= -1
        if axis_y:
            points[:, 0] *= -1
        return points

    @staticmethod
    def flip_boxes(boxes, axis_x, axis_y, locally=False):
        if axis_x:
            boxes[:, 1] *= -1 if not locally else 1
            boxes[:, 6] *= -1
            if boxes.size(-1) > 7:
                boxes[:, 8] *= -1
        if axis_y:
            boxes[:, 0] *= -1 if not locally else 1
            boxes[:, 6] = -boxes[:, 6] + np.pi
            if boxes.size(-1) > 7:
                boxes[:, 7] *= -1
        return boxes

    def params(self, data_dict):
        return {k: self.enable(self.axis_prob) if k in self.axis else False for k in ['x', 'y']}

    def backward(self, data_dict, x, y):
        self.forward(data_dict, x, y)

    def forward(self, data_dict, x, y):
        if 'points' in data_dict:
            data_dict['points'] = self.flip_points(data_dict['points'], x, y)
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = self.flip_boxes(data_dict['gt_boxes'], x, y)


@AUGMENTOR.register('global_sparsify')
class GlobalSparsify(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.ratio = abs(kwargs.get('keep_ratio', 0.0))

    def params(self, data_dict):
        points = data_dict['points']
        mask = self.enable(self.ratio, size=points.size(0))
        mask_inv = torch.logical_not(mask)
        drop_part = points[mask_inv]
        return dict(mask=mask, mask_inv=mask_inv, drop_part=drop_part)

    def backward(self, data_dict, mask, mask_inv, drop_part):
        points = data_dict['points']
        raw_points = points.new_zeros([mask.size(0), points.size(-1)])
        raw_points[mask] = points
        raw_points[mask_inv] = drop_part.to(points.device)
        data_dict['points'] = raw_points

    def forward(self, data_dict, mask, mask_inv, drop_part):
        data_dict['points'] = data_dict['points'][mask]


@AUGMENTOR.register('frustum_sparsify')
class FrustumSparsify(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.head = kwargs.get('direction', [-np.pi / 4, np.pi / 4])
        self.range = kwargs.get('range', np.pi / 4)
        self.ratio = np.clip(abs(kwargs.get('keep_ratio', 0.5)), 0, 1)

    def params(self, data_dict):
        if 'points' in data_dict:
            head = self.uniform(self.head[0], self.head[1])
            width = self.uniform(0.0, np.pi / 4)

            points = data_dict['points']
            point_direction = torch.arctan2(points[:, 1], points[:, 0])
            keep = self.angle_out_of_range(point_direction, head, width)
            remove = torch.logical_not(keep)
            keep[remove] = self.enable(self.ratio, size=remove.sum(), device=points.device)

            mask_inv = torch.logical_not(keep)
            drop_part = data_dict['points'][mask_inv]
            return dict(mask=keep, mask_inv=mask_inv, drop_part=drop_part)
        else:
            return dict(mask=None, mask_inv=None, drop_part=None)

    def backward(self, data_dict, mask, mask_inv, drop_part):
        points = data_dict['points']
        raw_points = points.new_zeros([mask.size(0), points.size(-1)])
        raw_points[mask] = points
        raw_points[mask_inv] = drop_part.to(points.device)
        data_dict['points'] = raw_points

    def forward(self, data_dict, mask, mask_inv, drop_part):
        if 'points' in data_dict:
            data_dict['points'] = data_dict['points'][mask]


@AUGMENTOR.register('frustum_noise', 'frustum_jitter')
class FrustumJitter(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.head = kwargs.get('direction', [-np.pi / 4, np.pi / 4])
        self.range = kwargs.get('range', np.pi / 4)
        self.std = abs(kwargs.get('std', 0.5))

    def params(self, data_dict):
        if 'points' in data_dict:
            head = self.uniform(self.head[0], self.head[1])
            width = self.uniform(0.0, np.pi / 4)

            points = data_dict['points']
            point_direction = torch.arctan2(points[:, 1], points[:, 0])
            out_range = self.angle_out_of_range(point_direction, head, width)
            in_range = torch.logical_not(out_range)

            loc_noise = self.normal(0.0, std=self.std, size=(in_range.sum(), 3))
            return dict(mask=in_range, loc_noise=loc_noise)
        else:
            return dict(mask=None, loc_noise=None)

    def backward(self, data_dict, mask, loc_noise):
        self.forward(data_dict, mask, -loc_noise)

    def forward(self, data_dict, mask, loc_noise):
        if 'points' in data_dict:
            points = data_dict['points']
            data_dict['points'][mask, :3] += loc_noise.to(points.device)


@AUGMENTOR.register('box_rotate', 'local_rotate')
class BoxRotate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [-np.pi / 4, np.pi / 4])
        assert len(self.range) == 2

    @staticmethod
    def rotate_boxes_local(boxes, rot):
        if boxes.size(0):
            boxes = GlobalRotate.rotate_boxes(boxes, rot, locally=True)
        return boxes

    @staticmethod
    def rotate_points_local(points, rots, masks, boxes):
        if boxes.size(0):
            for mask, box, rot in zip(masks, boxes, rots):
                offset = box[:3]
                points_of_box = points[mask]
                points_of_box[:, :3] -= offset
                points_of_box = GlobalRotate.rotate_points(points_of_box, rot)
                points_of_box[:, :3] += offset
                points[mask] = points_of_box
        return points

    def params(self, data_dict):
        if 'gt_boxes' in data_dict:
            masks = self.try_points_in_boxes_masks_from_cache(data_dict)
            rot_noise = self.uniform(self.range[0], self.range[1], size=data_dict['gt_boxes'].size(0))
            return dict(masks=masks, rot_noise=rot_noise)
        else:
            return dict(masks=None, rot_noise=None)

    def backward(self, data_dict, masks, rot_noise):
        self.forward(data_dict, masks, -rot_noise)

    def forward(self, data_dict, masks, rot_noise):
        if 'gt_boxes' in data_dict:
            boxes = data_dict['gt_boxes']
            data_dict['gt_boxes'] = self.rotate_boxes_local(boxes, rot_noise)
            if 'points' in data_dict:
                points = data_dict['points']
                data_dict['points'] = self.rotate_points_local(points, rot_noise, masks, boxes)


@AUGMENTOR.register('box_translate', 'local_translate')
class BoxTranslate(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.std = kwargs.get('std', [0, 0, 0])
        assert len(self.std) == 3

    @staticmethod
    def translate_boxes_local(boxes, trans):
        if boxes.size(0):
            boxes = GlobalTranslate.translate_boxes(boxes, trans)
        return boxes

    @staticmethod
    def translate_points_local(points, trans, masks, boxes):
        if boxes.size(0):
            offsets = torch.zeros_like(points[:, :3])
            for mask, offset in zip(masks, trans):
                offsets[mask] = offset.to(points.device)
            points[:, :3] += offsets
        return points

    def params(self, data_dict):
        if 'gt_boxes' in data_dict:
            masks = self.try_points_in_boxes_masks_from_cache(data_dict)
            num_boxes = data_dict['gt_boxes'].size(0)
            trans_noise = self.normal(0.0, [self.std] * num_boxes)
            return dict(masks=masks, trans_noise=trans_noise)
        else:
            return dict(masks=None, trans_noise=None)

    def backward(self, data_dict, masks, trans_noise):
        self.forward(data_dict, masks, -trans_noise)

    def forward(self, data_dict, masks, trans_noise):
        if 'gt_boxes' in data_dict:
            boxes = data_dict['gt_boxes']
            data_dict['gt_boxes'] = self.translate_boxes_local(boxes, trans_noise)
            if 'points' in data_dict:
                points = data_dict['points']
                data_dict['points'] = self.translate_points_local(points, trans_noise, masks, boxes)


@AUGMENTOR.register('box_scale', 'local_scale')
class BoxScale(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.range = kwargs.get('range', [0.95, 1.05])
        assert len(self.range) == 2

    @staticmethod
    def scale_boxes_local(boxes, scale):
        if boxes.size(0):
            boxes[:, 3:6] *= scale.to(boxes.device).unsqueeze(1)
        return boxes

    @staticmethod
    def scale_points_local(points, scales, masks, boxes):
        if boxes.size(0):
            for index, offset, scale in zip(masks, boxes[:, :3], scales):
                points_of_box = points[index]
                points_of_box[:, :3] -= offset
                points_of_box = GlobalScale.scale_points(points_of_box, scale)
                points_of_box[:, :3] += offset
                points[index] = points_of_box
        return points

    def params(self, data_dict):
        if 'gt_boxes' in data_dict:
            masks = self.try_points_in_boxes_masks_from_cache(data_dict)
            scale_noise = self.uniform(self.range[0], self.range[1], size=data_dict['gt_boxes'].size(0))
            return dict(masks=masks, scale_noise=scale_noise)
        else:
            return dict(masks=None, scale_noise=None)

    def backward(self, data_dict, masks, scale_noise):
        self.forward(data_dict, masks, 1.0 / scale_noise)

    def forward(self, data_dict, masks, scale_noise):
        if 'gt_boxes' in data_dict:
            boxes = data_dict['gt_boxes']
            data_dict['gt_boxes'] = self.scale_boxes_local(boxes, scale_noise)
            if 'points' in data_dict:
                points = data_dict['points']
                data_dict['points'] = self.scale_points_local(points, scale_noise, masks, boxes)


@AUGMENTOR.register('box_flip', 'local_flip')
class BoxFlip(Augmentor):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.axis = kwargs.get('axis', ['x'])
        self.axis_prob = kwargs.get('axis_prob', 0.5)

    @staticmethod
    def flip_points_local(points, axes_x, axes_y, masks, boxes):
        if boxes.size(0):
            for index, offset, a_x, a_y in zip(masks, boxes[:, :3], axes_x, axes_y):
                points_of_box = points[index]
                points_of_box[:, :3] -= offset
                points_of_box = GlobalFlip.flip_points(points_of_box, a_x, a_y)
                points_of_box[:, :3] += offset
                points[index] = points_of_box
        return points

    @staticmethod
    def flip_boxes_local(boxes, axes_x, axes_y):
        if boxes.size(0):
            for i in range(boxes.shape[0]):
                boxes[i:i + 1] = GlobalFlip.flip_boxes(boxes[i:i + 1], axes_x[i], axes_y[i], locally=True)
        return boxes

    def params(self, data_dict):
        if 'gt_boxes' in data_dict:
            num_boxes = data_dict['gt_boxes'].size(0)
            masks = self.try_points_in_boxes_masks_from_cache(data_dict)
            x = self.enable(self.axis_prob, size=num_boxes) \
                if 'x' in self.axis else torch.zeros([num_boxes], dtype=torch.bool)
            y = self.enable(self.axis_prob, size=num_boxes) \
                if 'y' in self.axis else torch.zeros([num_boxes], dtype=torch.bool)
            return dict(x=x, y=y, masks=masks)
        else:
            return dict(x=None, y=None, masks=None)

    def backward(self, data_dict, masks, x, y):
        self.forward(data_dict, masks, x, y)

    def forward(self, data_dict, masks, x, y):
        if 'gt_boxes' in data_dict:
            boxes = data_dict['gt_boxes']
            data_dict['gt_boxes'] = self.flip_boxes_local(boxes, x, y)
            if 'points' in data_dict:
                points = data_dict['points']
                data_dict['points'] = self.flip_points_local(points, x, y, masks, boxes)


# @AUGMENTS.register_module('part_drop')
# class PartitionDrop(object):
#     pass
#
#
# @AUGMENTS.register_module('part_swap')
# class PartitionSwap(object):
#     pass
#
#
# @AUGMENTS.register_module('part_noise')
# class PartitionNoise(object):
#     pass
#
#
# @AUGMENTS.register_module('part_mix')
# class PartitionMix(object):
#     pass
#
#
# @AUGMENTS.register_module('part_sparsify')
# class PartitionSparsify(object):
#     pass
#
# @AUGMENTS.register_module('box_swap', 'local_swap')
# class BoxSwap(object):
#     pass
# @AUGMENTOR.register_module('box_drop')
# class BoxDrop(Augmentor):
#     def __init__(self, kwargs):
#         super().__init__(kwargs)
#         self.drop_num = kwargs.get('num', {})
#
#     @staticmethod
#     def remove_points_in_boxes(points, boxes, enlarge=None):
#         from ...utils.box_utils import remove_points_in_boxes3d, enlarge_box3d
#         enlarge = [0.2, 0.2, 0.2] if enlarge is None else enlarge
#         enlarged_boxes = enlarge_box3d(boxes, enlarge)
#         points = remove_points_in_boxes3d(points, enlarged_boxes)
#         return points
#
#     def random(self, data_dict):
#         gt_names = data_dict['gt_names']
#         drop_indices = []
#         for cls, drop_num in self.drop_num.items():
#             cls_flags = cls == gt_names
#             cls_num = cls_flags.sum()
#             lower, upper = np.floor(drop_num), np.ceil(drop_num)
#             drop_num = int(np.random.choice([lower, upper], p=[upper - drop_num, drop_num - lower]))
#             if cls_num <= drop_num or drop_num == 0: continue
#             drop_ind = np.random.choice(np.where(cls_flags)[0], size=int(drop_num), replace=False)
#             drop_indices += list(drop_ind)
#         return dict(drop_which=drop_indices)
#
#     def forward(self, data_dict, drop_which):
#         if drop_which:
#             points, gt_boxes = data_dict['points'], data_dict['gt_boxes']
#             drop_boxes = gt_boxes[drop_which]
#             data_dict['points'] = self.remove_points_in_boxes(points, drop_boxes)


# @AUGMENTS.register_module('box_paste')
# class BoxPaste(object):
#     def __init__(self, kwargs):
#         from . import database_sampler
#         self.prob = kwargs.get('prob', 0.5)
#         self.verbose = kwargs.get('verbose', False)
#         self.db_sampler = database_sampler.DataBaseSampler(
#             sampler_cfg=kwargs['sampler_cfg'],
#             root_path=kwargs['root_dir'],
#             class_names=kwargs['class_name']
#         )
#
#     def __call__(self, data_dict):
#         self.db_sampler(data_dict)
#         return data_dict
#
# @AUGMENTS.register_module('background_swap')
# class BackgroundSwap(Transform):
#     def __init__(self, kwargs):
#         super().__init__(kwargs)
#         # we can rotate the background by 180 dg to approximate the background swap
#         # due to only the front 90 dg of scenes is labeled in kitti-like dataset.
#         self.fast_mode = kwargs.get('kitti', True)
#         self.dataset = kwargs.get('dataset', None)
#
#         assert self.fast_mode or self.dataset is None
#
#     def forward(self, data_dict):
#         from ...utils.box_utils import enlarge_box3d
#         from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
#         points = data_dict['points']
#         boxes = data_dict['gt_boxes']
#         coord = points[..., :3]
#         flags = (points_in_boxes_cpu(coord, enlarge_box3d(boxes, [0.2, 0.2, 0.2])) > 0).sum(-1)
#
#         if self.fast_mode:
#             pass
#
#         else:
#             pass
#


########################################################################################################################
class AugmentorList:
    def __init__(self, aug_list):
        self.augmentor_list = [AUGMENTOR.build_from_cfg(tf) for tf in aug_list]
        self.np_keys = None

    def tensor(self, data_dict):
        self.np_keys = {'points': False, 'gt_boxes': False}
        for k in self.np_keys:
            if k in data_dict and isinstance(data_dict[k], np.ndarray):
                self.np_keys[k] = True
                data_dict[k] = torch.from_numpy(data_dict[k])

    def numpy(self, data_dict):
        for k in self.np_keys:
            if self.np_keys[k]:
                data_dict[k] = data_dict[k].numpy()

    def __call__(self, data_dict, aug_logs={}):
        self.tensor(data_dict)
        for aug in self.augmentor_list:
            aug(data_dict, aug_logs.get(aug.name, None))
        self.numpy(data_dict)
        return data_dict

    def invert(self, data_dict, aug_logs={}):
        self.tensor(data_dict)
        for aug in self.augmentor_list[::-1]:
            aug.invert(data_dict, aug_logs.get(aug.name, None))
        self.numpy(data_dict)
        return data_dict
