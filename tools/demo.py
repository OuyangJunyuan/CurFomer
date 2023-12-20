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
        # seed: int = typer.Option(
        #     0,
        #     help="use this random seed"),
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
        for ind in (np.random.randint(0, len(dataset), arg.scenes) if arg.scene is None else [arg.scene]):
            batch_dict = dataset.collate_batch([dataset[ind]])
            dataset.load_data_to_gpu(batch_dict)

            pred_dicts, _ = model(batch_dict)

            points = batch_dict['points'][:, 1:5].view(-1, 4)
            gt_boxes = batch_dict['gt_boxes'][0] if 'gt_boxes' in batch_dict else None
            pred_boxes = pred_dicts[0]['pred_boxes'][:, :7].detach().view(-1, 7)
            pred_labels = pred_dicts[0]['pred_labels'].detach().view(-1)
            pred_scores = pred_dicts[0]['pred_scores'].detach().view(-1)

            logger.info("======")
            logger.info(f"scenes: {batch_dict['frame_id'][0]}")
            logger.info(f"points: {batch_dict['points'].shape}")
            logger.info(f"pred boxes: {pred_boxes.shape}")
            logger.info(f"gt boxes: {gt_boxes.shape if gt_boxes is not None else None}")

            gt_boxes = torch.cat((gt_boxes[:, :7], gt_boxes[:, -1:]), dim=-1) if gt_boxes is not None else None
            viz_scenes((None, gt_boxes), (points, pred_boxes), origin=True)

    logger.info("Done")


if __name__ == '__main__':
    mode = "train"
    app()
