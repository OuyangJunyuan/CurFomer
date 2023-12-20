from ..base import *
from ..logger import create_logger


class RunnerBase:
    state_keys = ['cur_loops', 'cur_epochs', 'cur_iters', 'mode', 'state']

    @Hook.begin_and_end
    def __init__(self, cfg, hooks, logger=None):
        self.cfg = cfg
        self.logger = logger if logger else create_logger(name="run")

        self._mode: str = self.cfg['mode']
        self.state: str = self.mode
        self.cur_loops = 0  # times we iterate dataloader
        self.cur_epochs = 0  # times we iterate dataloader when state is equal to mode
        self.cur_iters = 0  # times we call forward when state is equal to mode
        self.inner_iters = 0  # current sample id in dataloader
        self.work_id = 0  # current work id during traversal workflow

        Hook.insert_by_configs(hooks)
        Hook.insert_by_configs(self.cfg.get('custom_hooks', []))
        self.logger.info("\n%s" % Hook.info())

    @property
    def mode(self):
        return self._mode

    @property
    def training(self):
        return self.state == 'train'

    @property
    def non_training(self):
        return self.state == 'train'

    @property
    def one_epoch(self):
        return getattr(self, f'{self.state}_one_epoch')
    
    @property
    def workflow(self):
        def flatten(wf):
            return sum([[(w.state, w.split)] * w.epochs for w in wf], [])
        for work in flatten(self.cfg.workflows[self.mode]):
            yield work

    def state_dict(self):
        return {k: getattr(self, k, None) for k in self.state_keys}

    def load_state_dict(self, state_dict):
        state_dict = {k: v for k, v in state_dict.items() if k in self.state_keys}
        self.__dict__.update(**state_dict)
