import time

import torch
import pickle
from ..core.base import *


class Metric:
    def __init__(self, dataset, save_dir, metric_type, logger):
        self.metric_type = metric_type
        self.dataset = dataset
        self.save_dir = save_dir
        self.label_dir = save_dir / 'final_result/data' if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.label_dir:
            self.label_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.label_list = []

    @property
    def num_samples(self):
        return len(self.label_list)

    @property
    def num_objects(self):
        return sum([pred['name'].__len__() for pred in self.label_list])

    def add_batch(self, batch, pred_dicts):
        pred_labels = self.dataset.generate_prediction_dicts(
            batch, pred_dicts,
            self.dataset.class_names,
            output_path=self.label_dir
        )
        self.label_list.append(pred_labels)

    def compute(self):
        self.label_list = dist.all_gather(self.label_list, on_main=True, opt="insert")
        if self.label_list:
            self.label_list = sum(self.label_list, [])[:len(self.dataset)]
            self.logger.info(
                'average predicted number of objects(%d samples): %.1f' % (
                    self.num_samples, self.num_objects / max(1, self.num_samples)
                )
            )
            if self.save_dir:
                self.logger.info(f'Result is save to {self.save_dir}')
                pickle.dump(self.label_list, open(self.save_dir / 'result.pkl', 'wb'))

            try:
                ap_result_str, ap_dict = self.dataset.evaluation(
                    self.label_list, self.dataset.class_names,
                    eval_metric=self.metric_type,
                    output_path=self.save_dir
                )
                self.logger.info(ap_result_str)
                return ap_dict
            except:
                return None


class Recall:
    def __init__(self, thresh_list, logger):
        from collections import defaultdict
        self.thresh_list = thresh_list
        self.logger = logger

        self.infos = defaultdict(int)

    def add_batch(self, info_dict):
        self.infos['gt_num'] += info_dict.get('gt', 0)
        for key, num in info_dict.items():
            if key.startswith(('roi', 'rcnn')):
                self.infos[f'recall_{key}'] += num

    def compute(self):
        self.infos = dist.all_gather([self.infos], on_main=True)
        if self.infos:
            if dist.state.use_distributed:
                for key, val in self.infos[0].items():
                    for k in range(1, dist.state.num_processes):
                        self.infos[0][key] += self.infos[k][key]
            self.infos = self.infos[0]

            recall_dict = {}
            for key, num in self.infos.items():
                if key.startswith('recall'):
                    recall_dict[key] = num / max(self.infos['gt_num'], 1)
                    self.logger.info(f"{key}: {recall_dict[key]}")
            return recall_dict


class Latency:
    def __init__(self, skip=10):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.t1 = 0
        self.skip = skip

    def add_batch_before(self):
        self.t1 = time.time()

    def add_batch_after(self):
        if self.skip != 0:
            self.skip -= 1
        else:
            self.count += 1
            self.val = (time.time() - self.t1) * 1000
            self.sum += self.val
            self.avg = self.sum / self.count


class EvalHookImpl:
    def __init__(self, cfg):
        self.assign_state_handler_to_runner()
        self.root: Path = cfg.get('root', None)
        self.metric = None
        self.recall = None
        self.latency = None

    @staticmethod
    def assign_state_handler_to_runner():
        import warnings
        from numba.core.errors import NumbaPerformanceWarning
        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

        from ..core import DistributedRunner

        @Hook.begin_and_end
        @torch.no_grad()
        def test_one_epoch(self, *args, **kwargs):
            self.model.eval()
            self.batch_loop(*args, **kwargs)

        DistributedRunner.test_one_epoch = test_one_epoch

    @dist.state.on_main_process
    def send_batch_eval_info(self, run):
        min_iou = min(self.recall.thresh_list)
        ss = f"(%d,%d)/%d" % (self.recall.infos[f'recall_roi_{min_iou}'],
                              self.recall.infos[f'recall_rcnn_{min_iou}'],
                              self.recall.infos['gt_num'])

        rt = f"{self.latency.val:.2f}({self.latency.avg:.2f})"
        epoch_bar_dict = {'runtime': rt, f'recall_{min_iou}': ss}

        if hasattr(run, 'epoch_bar'):
            run.epoch_bar.update(epoch_bar_dict)
        if hasattr(run, 'tracker'):
            run.tracker.update(epoch_bar_dict)

    @dist.state.on_main_process
    def send_final_eval_info(self, run, recall_dict, ap_dict):
        if hasattr(run, 'tracker'):
            run.tracker.update(recall=recall_dict, eval_result=ap_dict)


@HOOKS.register()
class EvalHook(EvalHookImpl):
    priority = 1
    enable = True

    def before_test_one_epoch(self, run, model, dataloader):
        self.metric = Metric(dataloader.dataset, self.root / ('eval/epoch_%d' % run.cur_epochs),
                             run.unwrap_model.model_cfg.POST_PROCESSING.EVAL_METRIC, run.logger)
        self.recall = Recall(run.unwrap_model.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST, run.logger)
        self.latency = Latency()

    @dispatch
    def before_forward(self, run: when.state('test'), model, batch):
        self.latency.add_batch_before()

    @dispatch
    def after_forward(self, run: when.state('test'), model, batch):
        pred_dicts, info_dict = self.forward_output
        self.metric.add_batch(batch, pred_dicts)
        self.recall.add_batch(info_dict)
        self.latency.add_batch_after()
        self.send_batch_eval_info(run)

    def after_test_one_epoch(self, run, *args, **kwargs):
        recall_dict = self.recall.compute()
        ap_dict = self.metric.compute()
        self.send_final_eval_info(run, recall_dict, ap_dict)
