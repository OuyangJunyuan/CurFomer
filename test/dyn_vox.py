import os

import numpy as np
import torch
import accelerate

from rd3d.datasets import build_dataloader
from rd3d.core import Config
from rd3d import PROJECT_ROOT
from rd3d.models import build_detector
from tqdm import tqdm
from rd3d.utils import viz_utils
from matplotlib import pyplot as plt
from collections import defaultdict

os.chdir(PROJECT_ROOT)
acc = accelerate.Accelerator()
cfg = Config.fromfile_py("configs/voxformer/voxformer_4x2_80e_kitti_3cls.py")
# cfg.RUN.shuffle = False
dataloader = build_dataloader(cfg.DATASET, cfg.RUN, training=True)

model = build_detector(cfg.MODEL, dataset=dataloader.dataset).cuda()
while True:
    for batch_dict in tqdm(iterable=dataloader):
        dataloader.dataset.load_data_to_gpu(batch_dict)
        batch_dict = model.vfe(batch_dict)
        counts = batch_dict['voxel_features'][:, -1].int()
        assert (counts != 0).all()
        acc.wait_for_everyone()
