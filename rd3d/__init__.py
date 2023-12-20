from pathlib import Path
from easydict import EasyDict

from .core import (
    PROJECT_ROOT,
    Config,
    create_logger,
    dict_string,
    table_string,
    set_random_seed,
    DistributedRunner,
    RunnerBase
)
from .models import build_detector
from .datasets import build_dataloader
from .core.optimization import (
    build_optimizer,
    build_scheduler
)

__all__ = [
    'Path',
    'EasyDict',
    'PROJECT_ROOT',
    'build_detector',
    'build_dataloader',
    'build_optimizer',
    'build_scheduler',
    'Config',
    'create_logger',
    'dict_string',
    'table_string',
    'set_random_seed',
    'RunnerBase',
    'DistributedRunner'
]


def quick_demo(parser=None):
    import argparse
    import accelerate

    def add_args(arg_parser):
        arg_parser = argparse.ArgumentParser() if arg_parser is None else arg_parser
        arg_parser.add_argument('--cfg', type=str, default="configs/iassd/iassd_4x8_80e_kitti_3cls.py",
                                help='specify the config for training')
        arg_parser.add_argument('--ckpt', type=str, default=None,
                                help='checkpoint to start from')
        arg_parser.add_argument('--set', dest='set_cfgs', default=[], nargs='...',
                                help='set extra config keys if needed')
        arg_parser.add_argument('--seed', type=int, default=0,
                                help='random seed')
        arg_parser.add_argument('--batch', type=int, default=None,
                                help='batch_size')
        arg_parser.add_argument('--test', action='store_true', default=False)
        return arg_parser

    accelerator = accelerate.Accelerator()

    args = add_args(parser).parse_args()
    cfg = Config.merge_cmdline_setting(Config.fromfile(args.cfg), args.set_cfgs)
    set_random_seed(args.seed)

    cfg.RUN = cfg.get('RUN', EasyDict())
    cfg.RUN.seed = args.seed
    if 'samples_per_gpu' not in cfg.RUN:
        cfg.RUN.samples_per_gpu = cfg.RUN.workers_per_gpu = 4
    if args.batch:
        cfg.RUN.samples_per_gpu = args.batch

    dataloader = build_dataloader(cfg.DATASET, cfg.RUN, training=not args.test)
    if 'MODEL' in cfg:
        model = build_detector(cfg.MODEL, dataset=dataloader.dataset)
    else:
        model = None
    if args.ckpt:
        from rd3d.core.ckpt import load_from_file
        load_from_file(args.ckpt, model)
    return model, dataloader, cfg, args


def set_workspace():
    import os
    os.chdir(PROJECT_ROOT)
