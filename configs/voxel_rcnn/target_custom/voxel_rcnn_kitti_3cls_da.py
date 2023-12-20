from ...base.datasets.custom import *
from ...base.runtime.adam_onecycle_4_80e import *

DATASET.SWEEPS.RANGE = [0, 0]

MODEL = dict(
    NAME='VoxelRCNN',
    CLASS_NAMES=DATASET.CLASS_NAMES,
    VFE=dict(NAME='MeanVFE'),
    BACKBONE_3D=dict(NAME='VoxelBackBone8x'),
    MAP_TO_BEV=dict(NAME='HeightCompression', NUM_BEV_FEATURES=256),
    BACKBONE_2D=dict(NAME='BaseBEVBackbone',
                     LAYER_NUMS=[5, 5],
                     LAYER_STRIDES=[1, 2],
                     NUM_FILTERS=[64, 128],
                     UPSAMPLE_STRIDES=[1, 2],
                     NUM_UPSAMPLE_FILTERS=[128, 128]),
    DENSE_HEAD=dict(NAME='AnchorHeadSingle',
                    CLASS_AGNOSTIC=False,
                    USE_DIRECTION_CLASSIFIER=True,
                    DIR_OFFSET=0.78539,
                    DIR_LIMIT_OFFSET=0.0,
                    NUM_DIR_BINS=2,
                    ANCHOR_GENERATOR_CONFIG=[
                        {
                            'class_name': 'Vehicle',
                            'anchor_sizes': [[3.9, 1.6, 1.56]],
                            'anchor_rotations': [0, 1.57],
                            'anchor_bottom_heights': [-1.78],
                            'align_center': False,
                            'feature_map_stride': 8,
                            'matched_threshold': 0.6,
                            'unmatched_threshold': 0.45
                        },
                        {
                            'class_name': 'Pedestrian',
                            'anchor_sizes': [[0.8, 0.6, 1.73]],
                            'anchor_rotations': [0, 1.57],
                            'anchor_bottom_heights': [-0.6],
                            'align_center': False,
                            'feature_map_stride': 8,
                            'matched_threshold': 0.5,
                            'unmatched_threshold': 0.35
                        },
                        {
                            'class_name': 'Cyclist',
                            'anchor_sizes': [[1.76, 0.6, 1.73]],
                            'anchor_rotations': [0, 1.57],
                            'anchor_bottom_heights': [-0.6],
                            'align_center': False,
                            'feature_map_stride': 8,
                            'matched_threshold': 0.5,
                            'unmatched_threshold': 0.35
                        }
                    ],
                    TARGET_ASSIGNER_CONFIG=dict(NAME='AxisAlignedTargetAssigner',
                                                POS_FRACTION=-1.0,
                                                SAMPLE_SIZE=512,
                                                NORM_BY_NUM_EXAMPLES=False,
                                                MATCH_HEIGHT=False,
                                                BOX_CODER='ResidualCoder'),
                    LOSS_CONFIG=dict(
                        LOSS_WEIGHTS={
                            'cls_weight': 1.0,
                            'loc_weight': 2.0,
                            'dir_weight': 0.2,
                            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                        }
                    )),
    ROI_HEAD=dict(
        NAME='VoxelRCNNHead',
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
                USE_FAST_NMS=False,
                SCORE_THRESH=0.0,
                NMS_PRE_MAXSIZE=2048,
                NMS_POST_MAXSIZE=100,
                NMS_THRESH=0.7)
        ),
        ROI_GRID_POOL=dict(
            FEATURES_SOURCE=['x_conv2', 'x_conv3', 'x_conv4'],
            PRE_MLP=True,
            GRID_SIZE=6,
            POOL_LAYERS=dict(
                x_conv2=dict(MLPS=[[32, 32]],
                             QUERY_RANGES=[[4, 4, 4]],
                             POOL_RADIUS=[0.4],
                             NSAMPLE=[16],
                             POOL_METHOD='max_pool'),
                x_conv3=dict(MLPS=[[32, 32]],
                             QUERY_RANGES=[[4, 4, 4]],
                             POOL_RADIUS=[0.8],
                             NSAMPLE=[16],
                             POOL_METHOD='max_pool'),
                x_conv4=dict(MLPS=[[32, 32]],
                             QUERY_RANGES=[[4, 4, 4]],
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
                'rcnn_iou3d_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
        ),

    ),
    POST_PROCESSING=dict(
        RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
        SCORE_THRESH=0.1,
        OUTPUT_RAW_SCORE=False,

        EVAL_METRIC='kitti',

        NMS_CONFIG=dict(
            MULTI_CLASSES_NMS=False,
            NMS_TYPE='nms_gpu',
            NMS_THRESH=0.1,
            NMS_PRE_MAXSIZE=4096,
            NMS_POST_MAXSIZE=500)
    )
)
RUN.samples_per_gpu = 2
