from ..base.datasets.kitti_3cls import *
from ..base.runtime.adam_onecycle_4_80e import *

DATASET.DATA_PROCESSOR = [
    dict(NAME='mask_points_and_boxes_outside_range',
         REMOVE_OUTSIDE_BOXES=True),
    dict(NAME='shuffle_points',
         SHUFFLE_ENABLED={'train': True, 'test': False}),
    dict(NAME='transform_points_to_voxels_placeholder',
         VOXEL_SIZE=[0.25, 0.25, 0.8])
]
MODEL = dict(
    NAME='CenterPoint', CLASS_NAMES=DATASET.CLASS_NAMES,
    VFE=dict(NAME='DynamicVoxelVFE',
             WITH_DISTANCE=False,
             USE_ABSLOTE_XYZ=True,
             USE_NORM=True,
             NUM_FILTERS=[128, 128]),
    BACKBONE_3D=dict(NAME='CurveBackBone',
                     GROUP_SIZE=32,
                     BLOCKS=[[128, 128, 128],
                             [128, 128, 128],
                             [128, 128, 128],
                             [128, 128, 128]]),
    MAP_TO_BEV=dict(NAME='HeightCompressionMean', NUM_BEV_FEATURES=128),
    BACKBONE_2D=dict(NAME='BaseBEVBackbone',
                     LAYER_NUMS=[5, 5],
                     LAYER_STRIDES=[1, 2],
                     NUM_FILTERS=[64, 128],
                     UPSAMPLE_STRIDES=[1, 2],
                     NUM_UPSAMPLE_FILTERS=[256, 256]),
    DENSE_HEAD=dict(
        NAME='CenterHead',
        CLASS_AGNOSTIC=False,
        CLASS_NAMES_EACH_HEAD=[
            ['Car'],
            ['Pedestrian'],
            ['Cyclist']
        ],
        SHARED_CONV_CHANNEL=64,
        USE_BIAS_BEFORE_NORM=True,
        NUM_HM_CONV=2,
        SEPARATE_HEAD_CFG=dict(
            HEAD_ORDER=['center', 'center_z', 'dim', 'rot'],
            HEAD_DICT={
                'center': {'out_channels': 2, 'num_conv': 2},
                'center_z': {'out_channels': 1, 'num_conv': 2},
                'dim': {'out_channels': 3, 'num_conv': 2},
                'rot': {'out_channels': 2, 'num_conv': 2},
            }
        ),
        TARGET_ASSIGNER_CONFIG=dict(
            FEATURE_MAP_STRIDE=1,
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
            POST_CENTER_LIMIT_RANGE=[-75.2, -75.2, -2, 75.2, 75.2, 4],
            MAX_OBJ_PER_SAMPLE=500,
            NMS_CONFIG=dict(
                NMS_TYPE='nms_gpu',
                NMS_THRESH=0.01,
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
