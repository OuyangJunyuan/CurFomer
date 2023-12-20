import pickle
import numpy as np
import torch
from ..core.base import *
from ..utils.misc import replace_attr
from ..datasets.augmentor.transforms import AugmentorList
from ..utils import box_fusion_utils


class TestTimeAugHookImpl:
    def __init__(self, cfg):
        self.enable = cfg.enable
        self.tta_augs = []
        self.dataset = None
        self.fusion_cfg = EasyDict()

    def concatenate_prediction(self, pred_dicts_list):
        """
        Args:
            pred_dicts_list: (num_augs, batch, pred_dict)

        Returns:
            pred_dicts: (batch, {key: cat(pred_dict[key],...)})
        """
        num_augs = len(pred_dicts_list)
        batch_size = len(pred_dicts_list[0])
        keys = list(pred_dicts_list[0][0].keys())

        pred_dicts = []
        for bid in range(batch_size):
            pred_dicts.append(
                {
                    k: torch.cat([pred_dicts_list[aid][bid][k] for aid in range(num_augs)])
                    for k in keys
                }
            )
        return pred_dicts

    def boxes_fusion(self, model, pred_dicts, batch):
        recall_dict = {}
        final_pred_dicts = []
        for bid, pd in enumerate(pred_dicts):
            handler = getattr(box_fusion_utils, f'{self.fusion_cfg.NAME}_boxes_fusion_tta')
            final_boxes, final_scores, final_labels = handler(pd, self.fusion_cfg)

            recall_dict = model.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=bid, data_dict=batch,
                thresh_list=model.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST
            )
            final_pred_dicts.append({
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            })
        return final_pred_dicts, recall_dict


@HOOKS.register()
class TestTimeAugHook(TestTimeAugHookImpl):
    priority = 1
    enable = True

    @dispatch
    def before_test_one_epoch(self, run: when.state('test'), model, dataloader):
        self.dataset = dataloader.dataset
        tta_cfg = self.dataset.dataset_cfg.TEST_TIME_AUGMENTOR
        self.fusion_cfg = tta_cfg.BOXES_FUSION
        self.tta_augs = [AugmentorList([tta_cfg.TRANSFORMS[i] for i in tta_indices])
                         for tta_indices in tta_cfg.AUGMENTOR_QUEUE]

    @dispatch
    def after_forward(self, run: when.state('test'), model, batch):
        """
        Notes:
            1. if it's in online mode.
            2. collect test-time augmented data from dataset by given index in batch.
        """
        pred_dicts_list = [self.forward_output[0]]

        for tta_aug in self.tta_augs:
            with replace_attr(self.dataset, data_augmentor=tta_aug):
                batch_dict = self.dataset.collate_batch([self.dataset[i] for i in batch['index'].int().tolist()])
                self.dataset.load_data_to_gpu(batch_dict)
                pred_dicts, _ = model(batch_dict)
                for pd, aug_log in zip(pred_dicts, batch_dict['augment_logs']):
                    tta_aug.invert(dict(gt_boxes=pd['pred_boxes']), aug_log)
                pred_dicts_list.append(pred_dicts)

        pred_dicts = self.concatenate_prediction(pred_dicts_list)

        # from ..utils.viz_utils import viz_scenes
        # points = batch['points']
        # points = points[points[:, 0] == 0]
        # viz_scenes((points, pred_dicts[0]['pred_boxes']))

        self.forward_output[0], self.forward_output[1] = self.boxes_fusion(run.unwrap_model, pred_dicts, batch)

        # from ..utils.viz_utils import viz_scenes
        # points = batch['points']
        # points = points[points[:, 0] == 0]
        # viz_scenes((points, self.forward_output[0][0]['pred_boxes']))

        # with open('cache/boxes_fusion.pkl', 'wb') as f:
        #     boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        #     labels = pred_dicts[0]['pred_labels'].cpu().numpy()
        #     scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        #     pickle.dump(np.concatenate((boxes, scores[:, None], labels[:, None]), axis=-1), f)
        #     assert False
