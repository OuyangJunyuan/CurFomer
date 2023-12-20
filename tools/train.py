import os
from datetime import datetime
from typing import List, Union

import typer
import accelerate
from rd3d import *

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
        cfg_file: Path = typer.Option(
            ..., "--cfg",
            help="path to config file"),
        ckpt: Union[Path, None] = typer.Option(
            None,
            help="path to checkpoint file"),
        tag: str = typer.Option(
            "default",
            help="a tag to distinguish different experiments"),
        resume: bool = typer.Option(
            True,
            help="resume previous training"),
        pretrain: bool = typer.Option(
            False,
            help="pretrain model by the given checkpoint file"),
        seed: int = typer.Option(
            0,
            help="use this random seed"),
        batch_size: Union[int, None] = typer.Option(
            None, "--batch",
            help="overwrite the batch size in config file"),
        output: Path = typer.Option(
            PROJECT_ROOT / "output",
            help="path to output training information"),
        setting: Union[List[str], None] = typer.Option(
            None, "--set",
            help="overwrite the setting in config file"),
        level: str = typer.Option(
            "INFO",
            help="the message level of logging"
        ),
        tracker: List[str] = typer.Option(
            None,
            help="use wandb logger"
        )
):
    assert not (pretrain and ckpt is None), 'please pass ckpt by --ckpt for pretrain'

    """ update config from cmdline (arg) """
    arg = EasyDict(locals())
    cfg = Config.merge_cmdline_setting(Config.fromfile(cfg_file), setting)
    tag = EasyDict(dataset=cfg.DATASET.TYPE, model=cfg_file.stem, exp=tag, mode=mode)
    work_dir = output / '/'.join(tag.values())
    work_dir.mkdir(parents=True, exist_ok=True)

    """ update runtime config from cmdline (arg) """
    RUN = cfg.RUN
    RUN.tag = tag
    RUN.mode = mode
    RUN.seed = seed
    RUN.work_dir = work_dir
    if batch_size:
        RUN.samples_per_gpu = batch_size

    """ log training information """
    log_file = RUN.work_dir / ("logs/%s.txt" % datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = create_logger("train", scene=True, level=arg.level, file=log_file)
    logger.info(f"arg: {dict_string(arg)}\ncfg: {dict_string(cfg)}")
    with open(RUN.work_dir / "config.yaml", 'w') as f:
        f.write(f"arg: {dict_string(arg)}\ncfg: {dict_string(cfg)}")

    launch_experiment(arg, cfg, RUN, logger)


def launch_experiment(arg, cfg, RUN, logger):
    accelerator = accelerate.Accelerator(log_with=arg.tracker, project_dir=RUN.work_dir)

    set_random_seed(arg.seed)
    train_dataloader = build_dataloader(cfg.DATASET, RUN, training=True, logger=logger)
    test_dataloader = build_dataloader(cfg.DATASET, RUN, training=False, logger=logger)
    total_train_steps = len(train_dataloader) * RUN.max_epochs
    model = build_detector(cfg.MODEL, dataset=train_dataloader.dataset)
    optim = build_optimizer(cfg.OPTIMIZATION, model=model)
    sche = build_scheduler(cfg.LR, optimizer=optim, total_steps=total_train_steps)

    """ run experiment """
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    logger.info(f"TOTAL_BATCH_SIZE: {accelerator.num_processes * RUN.samples_per_gpu}")

    DistributedRunner(
        RUN, hooks=[
            dict(name='rd3d.hooks.train.TrainHook'),
            dict(name='rd3d.hooks.eval.EvalHook', root=RUN.work_dir),
            dict(name='rd3d.hooks.progress.ProgressBarHook'),
            dict(name='rd3d.hooks.checkpoint.CheckPointHook',
                 ckpt=arg.ckpt, pretrain=arg.pretrain, resume=arg.resume,
                 root=RUN.work_dir, **RUN.checkpoints),
            dict(name='rd3d.hooks.tracker.TrackerHook', interval=1, config=cfg)
        ],
        accelerator=accelerator,
        model=model, optim=optim, sche=sche,
        dataloaders=dict(train=train_dataloader, test=test_dataloader),
        logger=logger
    )

    logger.info("Done")


if __name__ == '__main__':
    mode = "train"
    app()
