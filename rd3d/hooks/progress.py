import tqdm
from ..core.base import *


class ProgressBarHookImpl:

    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_bar = None
        self.epoch_bar = None
        self.epoch_bar_remain_keys = ['metric']  # cfg.remain_keys

    def update_bar_display(self, run):
        self.batch_bar.set_postfix(total_it=run.cur_iters, **run.batch_bar)
        self.epoch_bar.set_postfix(**run.epoch_bar)

    def clear_bar_display(self, run):
        run.batch_bar = {}
        run.epoch_bar = {k: v for k, v in run.epoch_bar.items()
                         if k in self.epoch_bar_remain_keys}


@HOOKS.register()
class ProgressBarHook(ProgressBarHookImpl):
    enable = True

    def before_run(self, run):
        run.batch_bar = {}
        run.epoch_bar = {}
        self.enable = run.accelerator.is_main_process

    def before_epoch_loop(self, run):
        self.epoch_bar = tqdm.tqdm(total=run.cfg.max_epochs, desc='epochs',
                                   initial=run.cur_epochs, leave=True)

    def before_batch_loop(self, run, model, dataloader):
        self.batch_bar = tqdm.tqdm(total=len(dataloader), desc=run.state,
                                   leave=run.cur_epochs + 1 == run.cfg.max_epochs)

    def after_forward(self, run, *args, **kwargs):
        self.update_bar_display(run)
        self.batch_bar.update()

    def after_batch_loop(self, run, *args, **kwargs):
        if run.mode == run.state:
            self.epoch_bar.update()
        self.batch_bar.close()
        self.clear_bar_display(run)

    def after_epoch_loop(self, run):
        self.epoch_bar.close()
