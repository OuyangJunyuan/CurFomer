import logging
import os
from typing import List, Union

from ..core.base import *
from ..core import (
    create_logger,
    load_from_file,
    save_to_file,
)


class CheckPointHookImpl:
    CKPT_PATTERN = 'checkpoint_epoch_'

    def __init__(self, cfg):
        self.cfg = cfg
        self.ckpt: Union[Path, List[Path], None] = cfg.ckpt

        self.ckpt_dir: Path = Path(cfg.root) / "ckpt"
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.model_zoo: Path = Path(cfg.get("model_zoo", "model_zoo"))
        self.model_zoo.mkdir(exist_ok=True, parents=True)

        self.logger = create_logger(name="ckpt-hook", file=self.ckpt_dir / "log.txt", scene=False, level=logging.DEBUG)

    def ckpts_in_dir(self, fold):
        ckpt_list = list(fold.glob(f'*{self.CKPT_PATTERN}*.pth'))
        ckpt_list.sort(key=os.path.getmtime)
        return ckpt_list

    @dispatch
    def init_from_checkpoint(self, run: when.mode('train')):
        ws_ckpts = self.ckpts_in_dir(self.ckpt_dir)
        if ws_ckpts:
            if self.cfg.get("resume", False):
                self.ckpt = ws_ckpts[-1]
                load_from_file(filename=self.ckpt, runner=run, model=run.unwrap_model,
                               optimizer=run.optimizer, scheduler=run.scheduler)
                run.logger.info(f"resume from {self.ckpt}")
            else:
                raise FileExistsError('run existing experiment without resumption, please set a different --tag')
        else:
            if self.cfg.get('pretrain', False):
                if self.ckpt:
                    load_from_file(filename=self.ckpt, model=run.unwrap_model)
                    run.logger.info(f"pretrain from {self.ckpt}")
                else:
                    raise FileNotFoundError('pretrain without ckpt, please pass it by --ckpt')
            else:
                run.logger.info("train from scratch")

    @dispatch
    def init_from_checkpoint(self, run: when.mode('test')):
        if self.ckpt is None:
            ckpts = []
        elif self.ckpt.is_dir():
            ckpts = self.ckpts_in_dir(self.ckpt)
        else:
            ckpts = [self.ckpt]
        self.ckpt = ckpts
        run.cfg.max_epochs = len(self.ckpt)

    def save_this_ckpt(self, run):
        ckpt_name = self.ckpt_dir / f'{self.CKPT_PATTERN}{run.cur_epochs}.pth'
        save_to_file(
            filename=ckpt_name,
            runner=run,
            model=run.unwrap_model,
            optimizer=run.optimizer,
            scheduler=run.scheduler
        )
        self.logger.info(f"save {ckpt_name}")

    def remove_older_ckpt(self):
        ckpt_list = self.ckpts_in_dir(self.ckpt_dir)
        num_to_remove = len(ckpt_list) - self.cfg.get('max', 10)
        for i in range(num_to_remove):
            os.remove(ckpt_list[i])
            self.logger.info(f"remove {ckpt_list[i]}")


@HOOKS.register()
class CheckPointHook(CheckPointHookImpl):
    priority = 2

    def before_run(self, run):
        self.init_from_checkpoint(run)
        # self.init_metrics(run)

    @dist.state.on_main_process
    @dispatch
    def after_train_one_epoch(self, run: when.mode('train'), *args, **kwargs):
        self.save_this_ckpt(run)
        self.remove_older_ckpt()

    @dispatch
    def before_test_one_epoch(self, run: when.mode('test'), *args, **kwargs):
        load_from_file(filename=self.ckpt[run.cur_epochs], model=run.unwrap_model)
