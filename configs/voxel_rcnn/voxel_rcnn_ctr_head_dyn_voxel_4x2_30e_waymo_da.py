from ..base.datasets.waymo import *
from ..base.runtime.adam_onecycle_4_80e import *

RUN.max_epochs = 30
RUN.samples_per_gpu = 2
RUN.workflows.train = [dict(state='train', split='train', epochs=20),
                       dict(state='test', split='test', epochs=1),
                       dict(state='train', split='train', epochs=5)]
DATASET.SAMPLED_INTERVAL = {'train': 5, 'test': 5}
DATASET.POINT_FEATURE_ENCODING = dict(
    encoding_type='absolute_coordinates_encoding',
    used_feature_list=['x', 'y', 'z', 'intensity'],
    src_feature_list=['x', 'y', 'z', 'intensity', 'elongation'],
)
DATASET.DATA_PROCESSOR = [
    dict(NAME='mask_points_and_boxes_outside_range',
         REMOVE_OUTSIDE_BOXES=True),
    dict(NAME='shuffle_points',
         SHUFFLE_ENABLED=dict(train=True, test=True)),
    dict(NAME='transform_points_to_voxels_placeholder',
         VOXEL_SIZE=[0.1, 0.1, 0.15])
]

MODEL = dict(
    NAME='VoxelRCNN',
    CLASS_NAMES=DATASET.CLASS_NAMES,
    VFE=dict(NAME='DynMeanVFE'),
    BACKBONE_3D=dict(NAME='VoxelBackBone8x'),
    MAP_TO_BEV=dict(NAME='HeightCompression', NUM_BEV_FEATURES=256),
    BACKBONE_2D=dict(NAME='BaseBEVBackbone',
                     LAYER_NUMS=[5, 5],
                     LAYER_STRIDES=[1, 2],
                     NUM_FILTERS=[128, 256],
                     UPSAMPLE_STRIDES=[1, 2],
                     NUM_UPSAMPLE_FILTERS=[256, 256]),
    DENSE_HEAD=dict(NAME='CenterHead',
                    CLASS_AGNOSTIC=False,

                    CLASS_NAMES_EACH_HEAD=[
                        ['Vehicle', 'Pedestrian', 'Cyclist']
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
                        POST_CENTER_LIMIT_RANGE=[-75.2, -75.2, -2, 75.2, 75.2, 4],
                        MAX_OBJ_PER_SAMPLE=500,
                        NMS_CONFIG=dict(
                            NMS_TYPE='nms_gpu',
                            NMS_THRESH=0.7,
                            NMS_PRE_MAXSIZE=4096,
                            NMS_POST_MAXSIZE=500
                        )
                    )),
    ROI_HEAD=dict(NAME='VoxelRCNNHead',
                  CLASS_AGNOSTIC=True,

                  SHARED_FC=[256, 256],
                  CLS_FC=[256, 256],
                  REG_FC=[256, 256],
                  DP_RATIO=0.3,

                  NMS_CONFIG=dict(
                      TRAIN=dict(
                          NMS_TYPE='nms_gpu',
                          MULTI_CLASSES_NMS=False,
                          NMS_PRE_MAXSIZE=9000,
                          NMS_POST_MAXSIZE=512,
                          NMS_THRESH=0.8),
                      TEST=dict(
                          NMS_TYPE='nms_gpu',
                          MULTI_CLASSES_NMS=False,
                          NMS_PRE_MAXSIZE=1024,
                          NMS_POST_MAXSIZE=100,
                          NMS_THRESH=0.7)
                  ),
                  ROI_GRID_POOL=dict(
                      FEATURES_SOURCE=['x_conv2', 'x_conv3', 'x_conv4'],
                      PRE_MLP=True,
                      GRID_SIZE=6,
                      POOL_LAYERS=dict(
                          x_conv2=dict(MLPS=[[64, 64]],
                                       QUERY_RANGES=[[3, 3, 2]],
                                       POOL_RADIUS=[0.4],
                                       NSAMPLE=[16],
                                       POOL_METHOD='max_pool'),
                          x_conv3=dict(MLPS=[[64, 64]],
                                       QUERY_RANGES=[[3, 3, 2]],
                                       POOL_RADIUS=[0.8],
                                       NSAMPLE=[16],
                                       POOL_METHOD='max_pool'),
                          x_conv4=dict(MLPS=[[64, 64]],
                                       QUERY_RANGES=[[3, 3, 2]],
                                       POOL_RADIUS=[1.6],
                                       NSAMPLE=[16],
                                       POOL_METHOD='max_pool')
                      )
                  ),
                  TARGET_CONFIG=dict(
                      BOX_CODER='ResidualCoder',
                      ROI_PER_IMAGE=128,
                      FG_RATIO=0.5,

                      SAMPLE_ROI_BY_EACH_CLASS=True,
                      CLS_SCORE_TYPE='roi_iou',

                      CLS_FG_THRESH=0.75,
                      CLS_BG_THRESH=0.25,
                      CLS_BG_THRESH_LO=0.1,
                      HARD_BG_RATIO=0.8,
                      REG_FG_THRESH=0.55
                  ),
                  LOSS_CONFIG=dict(
                      CLS_LOSS='BinaryCrossEntropy',
                      REG_LOSS='smooth-l1',
                      CORNER_LOSS_REGULARIZATION=True,
                      GRID_3D_IOU_LOSS=False,
                      LOSS_WEIGHTS={
                          'rcnn_cls_weight': 1.0,
                          'rcnn_reg_weight': 1.0,
                          'rcnn_corner_weight': 1.0,
                          'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                      }
                  )),
    POST_PROCESSING=dict(
        RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
        SCORE_THRESH=0.1,
        OUTPUT_RAW_SCORE=False,

        EVAL_METRIC='kitti',

        NMS_CONFIG=dict(
            MULTI_CLASSES_NMS=False,
            NMS_TYPE='nms_gpu',
            NMS_THRESH=0.7,
            NMS_PRE_MAXSIZE=4096,
            NMS_POST_MAXSIZE=500)
    )
)
