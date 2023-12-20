import typer
import torch
import numpy as np
from typing import List, Union

import accelerate
from rd3d import *
from rd3d.core import ckpt
from rd3d.utils.viz_utils import viz_scenes

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
        cfg_file: Path = typer.Option(
            ..., "--cfg",
            help="path to config file"),
        ckpt: Union[Path, None] = typer.Option(
            None,
            help="path to checkpoint file"),
        setting: Union[List[str], None] = typer.Option(
            None, "--set",
            help="overwrite the setting in config file"),
        scene: int = typer.Option(
            None, "--scene"),
        scenes: int = typer.Option(
            1, "--scenes"),
):
    """ update config from cmdline (arg) """
    arg = EasyDict(locals())
    cfg = Config.merge_cmdline_setting(Config.fromfile(cfg_file), setting)
    logger = create_logger("train", scene=True)
    launch_experiment(arg, cfg, logger)


def launch_experiment(arg, cfg, logger):
    # accelerator = accelerate.Accelerator()
    # set_random_seed(arg.seed)
    dataset = build_dataloader(cfg.DATASET, training=False, logger=logger)
    model = build_detector(cfg.MODEL, dataset=dataset)

    ckpt.load_from_file(arg.ckpt, model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        # for ind in (np.random.randint(0, len(dataset), arg.scenes) if arg.scene is None else [arg.scene]):
        for ind in range(0, 1):
            batch_dict = dataset.collate_batch([dataset[ind]])
            dataset.load_data_to_gpu(batch_dict)
            model(batch_dict)
            with torch.profiler.profile(
                    # schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
                    # record_shapes=True,
                    # profile_memory=True,
                    # with_stack=True,
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
            ) as prof:
                for _ in range(10):
                    pred_dicts, _ = model(batch_dict)
            print(prof.key_averages().table(sort_by="self_cuda_time_total"))


if __name__ == '__main__':
    mode = "train"
    app()
