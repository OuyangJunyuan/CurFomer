from ..core.base import *


class TrainHookImpl:
    def __init__(self, cfg):
        self.cfg = cfg
        self.assign_state_handler_to_runner()

    @staticmethod
    def assign_state_handler_to_runner():
        @Hook.begin_and_end
        def train_one_epoch(self, *args, **kwargs):
            self.model.train()
            self.batch_loop(*args, **kwargs)

        from ..core import DistributedRunner
        DistributedRunner.train_one_epoch = train_one_epoch

    @staticmethod
    def step(run, pred_dict):
        grad_norm = 0
        loss = pred_dict['loss']
        lr = max(run.scheduler.get_last_lr())

        run.optimizer.zero_grad()
        run.accelerator.backward(loss)
        if run.accelerator.sync_gradients:
            grad_norm = run.accelerator.clip_grad_norm_(run.model.parameters(), run.cfg.grad_norm_clip)
        run.optimizer.step()
        run.scheduler.step()

        return loss.item(), lr, grad_norm.item()

    @staticmethod
    def send_training_info(run, train_infos, meta_infos):
        if hasattr(run, 'epoch_bar'):
            run.epoch_bar.update({k: '%.2e' % v for k, v in meta_infos.items()})

        if hasattr(run, 'tracker'):
            run.tracker.update(**{'meta/epoch': run.cur_epochs},
                               **{f'meta/{k}': meta_infos[k] for k in meta_infos},
                               **{f"train/{k}": train_infos[k] for k in train_infos})


@HOOKS.register()
class TrainHook(TrainHookImpl):
    priority = 1
    enable = True

    @dispatch
    def after_forward(self, run: when.state('train'), model, batch):
        pred_dict, info_dict = self.forward_output

        loss, lr, grad = self.step(run, pred_dict)

        self.send_training_info(
            run, train_infos=info_dict,
            meta_infos=dict(loss=loss, lr=lr, grad=grad)
        )
