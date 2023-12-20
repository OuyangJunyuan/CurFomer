from ..base.datasets.nuscenes_voxel import *
from ..base.runtime.adam_onecycle_2_20e import *

RUN.samples_per_gpu = 4
RUN.workers_per_gpu = 8
RUN.OPTIMIZATION.lr = RUN.LR.max_lr = 0.001
DATASET.VERSION = 'v1.0-mini'
DATASET.BALANCED_RESAMPLING = True
DATASET.PRED_VELOCITY = False
DATASET.DATA_PROCESSOR[2] = dict(NAME='transform_points_to_voxels',
                                 VOXEL_SIZE=[0.075, 0.075, 0.2],
                                 MAX_POINTS_PER_VOXEL=10,
                                 MAX_NUMBER_OF_VOXELS={'train': 120000, 'test': 160000})
DATASET.DATA_AUGMENTOR = dict(
    DISABLE_AUG_LIST=['placeholder'],
    AUG_CONFIG_LIST=[
        dict(NAME='gt_sampling',
             USE_ROAD_PLANE=True,
             DB_INFO_PATH=['nuscenes_dbinfos_10sweeps_withvelo.pkl'],
             PREPARE=dict(
                 filter_by_min_points=[
                     'car:5', 'truck:5', 'construction_vehicle:5', 'bus:5', 'trailer:5',
                     'barrier:5', 'motorcycle:5', 'bicycle:5', 'pedestrian:5', 'traffic_cone:5']
             ),
             SAMPLE_GROUPS=[
                 'car:2', 'truck:2', 'construction_vehicle:2', 'bus:2', 'trailer:2',
                 'barrier:2', 'motorcycle:2', 'bicycle:2', 'pedestrian:2', 'traffic_cone:2'],
             NUM_POINT_FEATURES=5,
             DATABASE_WITH_FAKELIDAR=False,
             REMOVE_EXTRA_WIDTH=[0.0, 0.0, 0.0],
             LIMIT_WHOLE_SCENE=True),
        dict(NAME='random_world_flip',
             ALONG_AXIS_LIST=['x', 'y']),
        dict(NAME='random_world_rotation',
             WORLD_ROT_ANGLE=[-0.78539816, 0.78539816]),
        dict(NAME='random_world_scaling',
             WORLD_SCALE_RANGE=[0.9, 1.1]),
        dict(NAME='random_world_translation',
             NOISE_TRANSLATE_STD=[0.5, 0.5, 0.5])
    ]
)

MODEL = dict(
    NAME='VoxelNeXt', CLASS_NAMES=DATASET.CLASS_NAMES,
    VFE=dict(NAME='MeanVFE'),
    BACKBONE_3D=dict(NAME='VoxelResBackBone8xVoxelNeXt'),
    DENSE_HEAD=dict(
        NAME='VoxelNeXtHead',
        CLASS_AGNOSTIC=False,
        INPUT_FEATURES=128,
        CLASS_NAMES_EACH_HEAD=[
            ['car'],
            ['truck', 'construction_vehicle'],
            ['bus', 'trailer'],
            ['barrier'],
            ['motorcycle', 'bicycle'],
            ['pedestrian', 'traffic_cone'],
        ],
        SHARED_CONV_CHANNEL=128,
        KERNEL_SIZE_HEAD=1,

        USE_BIAS_BEFORE_NORM=True,
        NUM_HM_CONV=2,
        SEPARATE_HEAD_CFG=dict(
            HEAD_ORDER=['center', 'center_z', 'dim', 'rot'],
            HEAD_DICT={
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2}
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
