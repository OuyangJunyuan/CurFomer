from ..core.base import *
from ..core import PROJECT_ROOT


class TrackerHookImpl:
    def __init__(self, cfg):
        self.interval = cfg.interval
        self.cfg = cfg.config
        self.tag = cfg.get('tag', cfg.config.RUN.tag)

    def wandb_init_kwargs(self, run):
        return dict(
            dir=self.cfg.RUN.work_dir,
            name='::'.join([self.tag.model, self.tag.exp]),
            resume=True,
            config=self.cfg,
            allow_val_change=True,
            tags=['%s[%s]' % (k, v) for k, v in self.tag.items()]
        )

    def wandb_init_tracker(self, run):
        import wandb

        if run.training:
            wandb.watch(models=run.unwrap_model, log='all', log_freq=len(run.dataloaders['train']) // 2)

    def tensorboard_init_tracker(self, run):
        if dist.state.is_main_process:
            log_dir = str(self.cfg.RUN.work_dir / PROJECT_ROOT.stem)
            print(f"tensorboard --logdir={log_dir} --port 7000")

    def log_train_info(self, run, info_dict, step):
        pass

    @dispatch
    def generate_train_log(self, run: when.state('train'), info_dict):
        return {
            'meta/epoch': run.cur_epochs,
            'meta/lr': run.tracker['lr'],
            'meta/loss': run.tracker['loss'],
            'meta/grad_norm': run.tracker['grad'],
            **{f"train/{key}": val for key, val in info_dict.items()}
        }

    def init_trackers(self, run):
        t_names = [t.value for t in run.accelerator.log_with]
        run.accelerator.init_trackers(
            project_name=PROJECT_ROOT.stem,
            init_kwargs={t: getattr(self, '%s_init_kwargs' % t, lambda *args: {})(run) for t in t_names}
        )
        for t in t_names:
            getattr(self, '%s_init_tracker' % t, lambda *args: None)(run)


@HOOKS.register()
class TrackerHook(TrackerHookImpl):
    """
    before_run: 模型和梯度记录频率设置
    after_forward: 每次迭代后跟踪数据到对应step
    after_test_one_epoch： 每轮验证后从EvalHook读取指标并记录到对于epoch
    """
    enable = True

    def before_run(self, run):
        run.tracker = {}
        self.enable = run.accelerator.is_main_process and run.accelerator.log_with
        self.init_trackers(run)

    def after_forward(self, run, model, batch):
        step = run.cur_iters

        if 0 == step % self.interval:
            run.accelerator.log(run.tracker, step=step)
