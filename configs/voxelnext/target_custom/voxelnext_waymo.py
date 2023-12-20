from ...base.datasets.custom import *
from ...base.runtime.adam_onecycle_4_80e import *

DATASET.SWEEPS.RANGE = [0, 0]
DATASET.POINT_FEATURE_ENCODING.used_feature_list = ['x', 'y', 'z', 'intensity']
DATASET.DATA_PROCESSOR[2] = dict(NAME='transform_points_to_voxels',
                                 VOXEL_SIZE=[0.1, 0.1, 0.15],
                                 MAX_POINTS_PER_VOXEL=5,
                                 MAX_NUMBER_OF_VOXELS={'train': 150000, 'test': 150000})

MODEL = dict(
    NAME='VoxelNeXt', CLASS_NAMES=DATASET.CLASS_NAMES,
    VFE=dict(NAME='MeanVFE'),
    BACKBONE_3D=dict(NAME='VoxelResBackBone8xVoxelNeXt',
                     SPCONV_KERNEL_SIZES=[5, 5, 3, 3],
                     OUT_CHANNEL=256,
                     CHANNELS=[32, 64, 128, 256, 256]),
    DENSE_HEAD=dict(
        NAME='VoxelNeXtHead',
        CLASS_AGNOSTIC=False,
        IOU_BRANCH=True,
        INPUT_FEATURES=256,

        CLASS_NAMES_EACH_HEAD=[
            ['Vehicle', 'Pedestrian', 'Cyclist']
        ],
        SHARED_CONV_CHANNEL=256,
        USE_BIAS_BEFORE_NORM=True,
        NUM_HM_CONV=2,
        SEPARATE_HEAD_CFG=dict(
            HEAD_ORDER=['center', 'center_z', 'dim', 'rot'],
            HEAD_DICT={
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
                'iou': {'out_channels': 1, 'num_conv': 2},
            }
        ),
        RECTIFIER=[0.68, 0.71, 0.65],
        TARGET_ASSIGNER_CONFIG=dict(
            FEATURE_MAP_STRIDE=8,
            NUM_MAX_OBJS=500,
            GAUSSIAN_OVERLAP=0.1,
            MIN_RADIUS=2
        ),
        LOSS_CONFIG=dict(
            LOSS_WEIGHTS={
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
        ),
        POST_PROCESSING=dict(
            SCORE_THRESH=0.1,
            POST_CENTER_LIMIT_RANGE=DATASET.POINT_CLOUD_RANGE,
            MAX_OBJ_PER_SAMPLE=500,
            NMS_CONFIG=dict(
                NMS_TYPE='nms_gpu',
                NMS_THRESH=[0.8, 0.55, 0.55],
                NMS_PRE_MAXSIZE=[2048, 1024, 1024],
                NMS_POST_MAXSIZE=[200, 150, 150]
            )
        )
    ),
    POST_PROCESSING=dict(
        RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
        EVAL_METRIC='kitti'
    )
)
