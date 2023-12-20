from ..base.datasets.nuscenes_voxel import *
from ..base.runtime.adam_onecycle_2_20e import *

DATASET.VERSION = 'v1.0-mini'
MODEL = dict(
    NAME='CenterPoint', CLASS_NAMES=DATASET.CLASS_NAMES,
    VFE=dict(NAME='MeanVFE'),
    BACKBONE_3D=dict(NAME='VoxelResBackBone8x'),
    MAP_TO_BEV=dict(NAME='HeightCompression', NUM_BEV_FEATURES=256),
    BACKBONE_2D=dict(NAME='BaseBEVBackbone',
                     LAYER_NUMS=[5, 5],
                     LAYER_STRIDES=[1, 2],
                     NUM_FILTERS=[128, 256],
                     UPSAMPLE_STRIDES=[1, 2],
                     NUM_UPSAMPLE_FILTERS=[256, 256]),
    DENSE_HEAD=dict(
        NAME='CenterHead',
        CLASS_AGNOSTIC=False,
        CLASS_NAMES_EACH_HEAD=[
            ['car'],
            ['truck', 'construction_vehicle'],
            ['bus', 'trailer'],
            ['barrier'],
            ['motorcycle', 'bicycle'],
            ['pedestrian', 'traffic_cone'],
        ],
        SHARED_CONV_CHANNEL=64,
        USE_BIAS_BEFORE_NORM=True,
        NUM_HM_CONV=2,
        SEPARATE_HEAD_CFG=dict(
            HEAD_ORDER=['center', 'center_z', 'dim', 'rot', 'vel'],
            HEAD_DICT={
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
                'vel': {'out_channels': 2, 'num_conv': 2},
            }
        ),
        TARGET_ASSIGNER_CONFIG=dict(
            FEATURE_MAP_STRIDE=8,
            NUM_MAX_OBJS=500,
            GAUSSIAN_OVERLAP=0.1,
            MIN_RADIUS=2
        ),
        LOSS_CONFIG=dict(
            LOSS_WEIGHTS={
                'cls_weight': 1.0,
                'loc_weight': 0.25,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]
            }
        ),
        POST_PROCESSING=dict(
            SCORE_THRESH=0.1,
            POST_CENTER_LIMIT_RANGE=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            MAX_OBJ_PER_SAMPLE=500,
            NMS_CONFIG=dict(
                NMS_TYPE='nms_gpu',
                NMS_THRESH=0.2,
                NMS_PRE_MAXSIZE=4096,
                NMS_POST_MAXSIZE=500
            )
        )
    ),
    POST_PROCESSING=dict(
        RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
        EVAL_METRIC='kitti'
    )
)

RUN.samples_per_gpu = 4
RUN.workers_per_gpu = 8
