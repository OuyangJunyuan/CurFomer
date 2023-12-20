from ..core.base import *
from ..utils.misc import replace_attr
from ..datasets.augmentor.transforms import AugmentorList


class TestTimeAugHookImpl:
    def __init__(self, cfg):
        self.enable = cfg.enable
        self.tta_augs = None


@HOOKS.register()
class TestTimeAugHook(TestTimeAugHookImpl):
    priority = 1
    enable = True

    @dispatch
    def before_test_one_epoch(self, run: when.state('test'), model, dataloader):
        tta_cfg = dataloader.dataset.dataset_cfg.TEST_TIME_AUGMENTOR
        assert tta_cfg.get('ENABLE_INDEX', -1) != -1
        tta_augs = AugmentorList([tta_cfg.TRANSFORMS[i] for i in tta_cfg.AUGMENTOR_QUEUE[tta_cfg.ENABLE_INDEX]])
        self.tta_augs = replace_attr(dataloader.dataset, data_augmentor=tta_augs).__enter__()

    @dispatch
    def after_forward(self, run: when.state('test'), model, batch):
        for pd, aug_log in zip(self.forward_output[0], batch['augment_logs']):
            self.tta_augs.obj.data_augmentor.invert(dict(gt_boxes=pd['pred_boxes']), aug_log)

    @dispatch
    def after_test_one_epoch(self, run: when.state('test'), model, dataloader):
        self.tta_augs.__exit__()
