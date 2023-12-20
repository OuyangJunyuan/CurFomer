from .base_runner import RunnerBase, Hook
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from typing import Dict


class DistributedRunner(RunnerBase):
    def __init__(self, cfg, hooks, accelerator, model, dataloaders, optim=None, sche=None, logger=None):
        super().__init__(cfg, hooks, logger)
        self.accelerator: Accelerator = accelerator
        self.model: DistributedDataParallel = self.accelerator.prepare(model)
        self.optimizer: AcceleratedOptimizer = self.accelerator.prepare(optim)
        self.scheduler: AcceleratedScheduler = self.accelerator.prepare(sche)
        self.dataloaders: Dict[str, DataLoader] = {k: self.accelerator.prepare(v) for k, v in dataloaders.items()}
        self.run()

    @property
    def unwrap_model(self):
        return self.accelerator.unwrap_model(self.model)

    @Hook.begin_and_end
    def forward(self, model, batch_dict):
        pred_ret_dict, ext_info_dict = model(batch_dict)
        self.cur_iters += self.mode == self.state
        return pred_ret_dict, ext_info_dict

    @Hook.begin_and_end
    def batch_loop(self, model, dataloader):
        for self.inner_iters, batch_dict in enumerate(dataloader):
            dataloader.dataset.load_data_to_gpu(batch_dict)
            self.forward(model, batch_dict)
        self.cur_loops += 1
        self.cur_epochs += self.mode == self.state

    @Hook.begin_and_end
    def epoch_loop(self):
        while self.cur_epochs < self.cfg.max_epochs:
            for self.state, split in self.workflow:
                self.work_id += 1

                if self.cur_loops >= self.work_id:  # skip for
                    self.logger.info(f"work {self.work_id} [state({self.state}) data({split})] done")
                    continue
                if self.cur_epochs >= self.cfg.max_epochs:
                    self.logger.warning("jump out works because the max_epochs has been reached")
                    break

                self.one_epoch(self.model, self.dataloaders[split])

    @Hook.begin_and_end
    def run(self):
        self.logger.info(f"running {'/'.join(self.cfg.tag.values())} ...")
        self.epoch_loop()
