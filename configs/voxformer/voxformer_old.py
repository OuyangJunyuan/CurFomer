from ..base.datasets.kitti_3cls import *
from ..base.runtime.adam_onecycle_8_80e import *

DATASET.DATA_PROCESSOR = [
    dict(NAME='mask_points_and_boxes_outside_range',
         REMOVE_OUTSIDE_BOXES=True),
    dict(NAME='shuffle_points',
         SHUFFLE_ENABLED={'train': True, 'test': False}),
    dict(NAME='transform_points_to_voxels_placeholder',
         VOXEL_SIZE=[0.25, 0.25, 0.2])
]

MODEL = dict(
    NAME='VoxFormer',
    CLASS_NAMES=DATASET.CLASS_NAMES,

    VFE=dict(NAME='DynamicVoxelVFE',
             WITH_DISTANCE=False,
             USE_ABSLOTE_XYZ=True,
             USE_NORM=True,
             NUM_FILTERS=[64, 128]),
    # VFE=dict(NAME='DynamicMeanVFEV2'),

    BACKBONE_3D=dict(NAME='CurveBackBone',
                     MULTI_GROUP=['reverse_coors',
                                  'flip_coors',
                                  'shift'][0],
                     GROUP_SIZE=128,
                     MULTI_SCALE_STRIDES=[4],
                     BLOCKS=[dict(name='PointMLP', pre=[128] * 2, pos=[128] * 1, ),
                             dict(name='PointMLP', pre=[128] * 2, pos=[128] * 1, ),
                             dict(name='PointMLP', pre=[256] * 2, pos=[256] * 1, ),
                             dict(name='PointMLP', pre=[256] * 2, pos=[256] * 1, ), ], ),

    DENSE_HEAD=dict(NAME='PointSegHead',
                    INPUT_FEATURES=256,
                    CLS_FC=[256, 256],
                    # SAMPLE_TOPK=256,
                    TARGET=dict(
                        # EXTRA_WIDTH=[0.5, 0.5, 0.5]
                    ),
                    LOSS=dict(WEIGHT=1.0, CENTERNESS=True, LOG_NAME='dense_head', )),

    # POINT_HEAD=dict(
    #     NAME='PointHeadVotePlus',
    #     ENABLE_TRAINING=False,
    #     CLASS_AGNOSTIC=False,
    #     VOTE_SAMPLER=dict(name='select', range=[0, 256], sample=[0, 256]),
    #     VOTE_MODULE=
    #     [
    #         dict(mlps=[128],
    #              max_translation_range=[3.0, 3.0, 2.0],
    #              sa=dict(groupers=[dict(name='ball', query=dict(radius=4.8, neighbour=16), mlps=[256, 256, 512]),
    #                                dict(name='ball', query=dict(radius=6.4, neighbour=32), mlps=[256, 512, 1024])],
    #                      aggregation=dict(name='cat-mlps')),
    #              train=dict(target={'set_ignore_flag': False, 'extra_width': [1.0, 1.0, 1.0]},
    #                         loss={'weight': 1.0, 'tb_tag': 'vote_reg_loss'}))
    #     ],
    #     SHARED_FC=[512],
    #     CLS_FC=[256, 256],
    #     REG_FC=[256, 256],
    #     BOX_CODER=dict(name='PointBinResidualCoder', angle_bin_num=12,
    #                    use_mean_size=True, mean_size=[[3.9, 1.6, 1.56],
    #                                                   [0.8, 0.6, 1.73],
    #                                                   [1.76, 0.6, 1.73]]),
    #     TARGET_CONFIG={'method': 'mask', 'gt_central_radius': False, 'extra_width': [0.2, 0.2, 0.2]},
    #     LOSS_CONFIG=dict(LOSS_CLS='WeightedBinaryCrossEntropyLoss',
    #                      LOSS_REG='WeightedSmoothL1Loss',
    #                      AXIS_ALIGNED_IOU_LOSS_REGULARIZATION=False,
    #                      CORNER_LOSS_REGULARIZATION=False,
    #                      WEIGHTS={'point_cls_weight': 1.0,
    #                               'point_offset_reg_weight': 1.0,
    #                               'point_angle_cls_weight': 0.2,
    #                               'point_angle_reg_weight': 1.0,
    #                               'point_iou_weight': 1.0,
    #                               'point_corner_weight': 1.0}),
    # ),
    # POST_PROCESSING=dict(
    #     EVAL_METRIC='kitti',
    #     SCORE_THRESH=0.1,
    #     OUTPUT_RAW_SCORE=False,
    #     RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
    #     NMS_CONFIG=dict(NMS_TYPE='nms_gpu',
    #                     NMS_THRESH=0.01,
    #                     NMS_PRE_MAXSIZE=4096,
    #                     NMS_POST_MAXSIZE=500,
    #                     MULTI_CLASSES_NMS=False, )
    # )
)

RUN.workflows = dict(
    train=[dict(state='train', split='train', epochs=80)],
    test=[dict(state='test', split='test', epochs=1)]
)
