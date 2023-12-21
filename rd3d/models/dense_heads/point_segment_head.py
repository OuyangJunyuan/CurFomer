import torch
from easydict import EasyDict
from ..model_utils import build_mlps
from ...utils import common_utils, box_utils, loss_utils
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


class PointSegmentor(torch.nn.Module):
    def __init__(self, input_channels, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = EasyDict(model_cfg)
        self.num_class = num_class
        self.seg_mlps = build_mlps(self.model_cfg.CLS_FC,
                                   in_channels=input_channels,
                                   out_channels=self.num_class)
        self.loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        self.train_dict = {}
        self.init_weights()

    def init_weights(self, weight_init='xavier'):
        init_func = {'kaiming': torch.nn.init.kaiming_normal_,
                     'xavier': torch.nn.init.xavier_normal_,
                     'normal': torch.nn.init.normal_}[weight_init]

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.01)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        boxes_list = batch_dict['gt_boxes_list']
        coors_list = torch.split(self.train_dict['coors'],
                                 self.train_dict['numbs'].tolist(), dim=0)

        cls_labels = []
        box_labels = []
        for b in range(batch_size):
            this_boxes = boxes_list[b]
            this_coors = coors_list[b][:, -3:]
            this_boxes_cls = this_boxes[:, -1:]
            this_boxes = this_boxes[:, :7]

            in_box_id = points_in_boxes_gpu(this_coors[None], this_boxes[None])[0].long()
            background = in_box_id < 0

            this_cls_labels = this_boxes_cls[in_box_id]
            this_cls_labels[background] = 0
            this_box_labels = this_boxes[in_box_id]
            this_box_labels[background] = 0

            if self.model_cfg.TARGET.get('EXTRA_WIDTH', False):
                this_enlarged_boxes = box_utils.enlarge_box3d(this_boxes, self.model_cfg.TARGET.EXTRA_WIDTH)
                in_enlarged_box_id = points_in_boxes_gpu(this_coors[None], this_enlarged_boxes[None])[0].long()
                enlarged_background = in_enlarged_box_id < 0
                ignored = background ^ enlarged_background
                this_cls_labels[ignored] = -1

            box_labels.append(this_box_labels)
        cls_labels = torch.cat(cls_labels, dim=0)
        box_labels = torch.cat(box_labels, dim=0)

        self.train_dict.update(cls_labels=cls_labels, box_labels=box_labels)

    @staticmethod
    def generate_centerness_label(points, boxes, mask, epsilon=1e-6,
                                  centerness_min=0.0, centerness_max=1.0):
        centerness = boxes.new_zeros(mask.size(0))

        boxes = boxes[mask]
        assert (boxes.sum(dim=-1) > 0).all()
        canonical_xyz = points[mask, :] - boxes[:, :3]
        rys = boxes[:, -1]
        canonical_xyz = common_utils.rotate_points_along_z(
            canonical_xyz.unsqueeze(dim=1), -rys
        ).squeeze(dim=1)

        distance_front = boxes[:, 3] / 2 - canonical_xyz[:, 0]
        distance_back = boxes[:, 3] / 2 + canonical_xyz[:, 0]
        distance_left = boxes[:, 4] / 2 - canonical_xyz[:, 1]
        distance_right = boxes[:, 4] / 2 + canonical_xyz[:, 1]
        distance_top = boxes[:, 5] / 2 - canonical_xyz[:, 2]
        distance_bottom = boxes[:, 5] / 2 + canonical_xyz[:, 2]

        centerness_l = torch.min(distance_front, distance_back) / (torch.max(distance_front, distance_back) + 1e-5)
        centerness_w = torch.min(distance_left, distance_right) / (torch.max(distance_left, distance_right) + 1e-5)
        centerness_h = torch.min(distance_top, distance_bottom) / (torch.max(distance_top, distance_bottom) + 1e-5)
        centerness_pos = torch.clamp(centerness_l * centerness_w * centerness_h, min=epsilon) ** (1 / 3.0)

        centerness[mask] = centerness_pos
        centerness = centerness_min + (centerness_max - centerness_min) * centerness
        assert not torch.isnan(centerness).any()
        assert not torch.isinf(centerness).any()
        return centerness

    def get_loss(self, tb_dict):
        coors = self.train_dict.pop('coors')
        num_coors = coors.size(0)
        cls_labels = self.train_dict.pop('cls_labels').view(num_coors)
        cls_logits = self.train_dict.pop('logits').view(num_coors, self.num_class)
        box_labels = self.train_dict.pop('box_labels').view(num_coors, -1)

        noignores = cls_labels >= 0
        positives = cls_labels > 0
        negatives = cls_labels == 0
        cls_weights = positives * 1.0 + negatives * 1.0
        cls_weights /= torch.clamp(positives.sum().float(), min=1.0)

        onehot = cls_logits.new_zeros((num_coors, self.num_class + 1))
        onehot[noignores, cls_labels[noignores].long()] = 1.0
        if self.model_cfg.LOSS.CENTERNESS:
            centerness = self.generate_centerness_label(coors, box_labels, positives)
            onehot *= centerness.view(-1, 1)

        cls_loss = self.loss_func(cls_logits, onehot[:, 1:], weights=cls_weights).mean(dim=-1).sum()
        cls_loss = self.model_cfg.LOSS.WEIGHT * cls_loss
        assert not (torch.isnan(cls_loss).any() or torch.isinf(cls_loss).any())
        assert not (torch.isnan(cls_logits).any() or torch.isinf(cls_logits).any())
        assert torch.logical_and(cls_labels[noignores] >= 1, cls_labels[noignores] <= self.num_class).all()

        if 'LOG_NAME' in self.model_cfg.LOSS:
            tb_dict.update({self.model_cfg.LOSS.LOG_NAME: cls_loss.item()})
        return cls_loss, tb_dict

    def forward(self, batch_dict):
        numbs = batch_dict['voxel_numbers']
        coors = batch_dict['point_coords']
        feats = batch_dict['voxel_features']

        cls_logits = self.seg_mlps(feats)

        if self.model_cfg.get('SAMPLE_TOPK', False):
            bs = batch_dict['batch_size']
            samples = self.model_cfg.SAMPLE_TOPK
            split_list = numbs.tolist()
            cls_scores = cls_logits.sigmoid().max(dim=-1)[0]
            select_indices = []
            batch_begin_index = 0
            cls_scores_list = cls_scores.split(split_list, dim=0)
            for bid in range(bs):
                select_indices.append(batch_begin_index + cls_scores_list[bid].topk(samples, dim=0)[1])
                batch_begin_index += numbs[bid]
            select_indices = torch.cat(select_indices, dim=0)
            batch_dict['point_coords'] = coors[select_indices].view(bs, samples, -1)
            batch_dict['point_features'] = feats[select_indices].view(bs, samples, -1)
            batch_dict['point_batch_id'] = batch_dict['voxel_coords'][select_indices, 0].view(bs, samples, 1)

        if self.training:
            self.train_dict.update(numbs=numbs, coors=coors, logits=cls_logits)
        batch_dict['cls_logits'] = cls_logits
        return batch_dict
