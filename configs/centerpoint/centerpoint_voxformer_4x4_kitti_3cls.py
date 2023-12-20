from ..base.datasets.kitti_3cls import *
from ..base.runtime.adam_onecycle_4_80e import *

MODEL = dict(
    NAME='PointVote', CLASS_NAMES=DATASET.CLASS_NAMES,
    VFE=dict(NAME='MeanVFE'),
    BACKBONE_3D=dict(NAME='VoxFormer',
                     MLPS=[[16, 16],
                           [32, 32],
                           [64, 64],
                           [128, 128]],
                     MAX_NUMBER_OF_VOXELS=40000,
                     GROUP_SIZE=64,
                     SAMPLER=dict(name='f-fps', sample=1024, gamma=1.0)),
    POINT_HEAD=dict(
        NAME='PointHeadVotePlus',
        CLASS_AGNOSTIC=False,
        VOTE_SAMPLER=dict(name='ctr', range=[0, 1024], sample=256, mlps=[256], class_names=DATASET.CLASS_NAMES,
                          train=dict(target={'extra_width': [0.5, 0.5, 0.5]},
                                     loss={'weight': 1.0, 'tb_tag': 'sasa_2'})),
        VOTE_MODULE=
        [
            dict(mlps=[128],
                 max_translation_range=[3.0, 3.0, 2.0],
                 sa=dict(groupers=[dict(name='ball', query=dict(radius=4.8, neighbour=16), mlps=[256, 256, 512]),
                                   dict(name='ball', query=dict(radius=6.4, neighbour=32), mlps=[256, 512, 1024])],
                         aggregation=dict(name='cat-mlps')),
                 train=dict(target={'set_ignore_flag': False, 'extra_width': [1.0, 1.0, 1.0]},
                            loss={'weight': 1.0, 'tb_tag': 'vote_reg_loss'}))
        ],
        SHARED_FC=[512],
        CLS_FC=[256, 256],
        REG_FC=[256, 256],
        BOX_CODER=dict(name='PointBinResidualCoder', angle_bin_num=12,
                       use_mean_size=True, mean_size=[[3.9, 1.6, 1.56],
                                                      [0.8, 0.6, 1.73],
                                                      [1.76, 0.6, 1.73]]),
        TARGET_CONFIG={'method': 'mask', 'gt_central_radius': False, 'extra_width': [0.2, 0.2, 0.2]},
        LOSS_CONFIG=dict(LOSS_CLS='WeightedBinaryCrossEntropyLoss',
                         LOSS_REG='WeightedSmoothL1Loss',
                         AXIS_ALIGNED_IOU_LOSS_REGULARIZATION=True,
                         CORNER_LOSS_REGULARIZATION=True,
                         WEIGHTS={'point_cls_weight': 1.0,
                                  'point_offset_reg_weight': 1.0,
                                  'point_angle_cls_weight': 0.2,
                                  'point_angle_reg_weight': 1.0,
                                  'point_iou_weight': 1.0,
                                  'point_corner_weight': 1.0}),
    ),
    POST_PROCESSING=dict(
        EVAL_METRIC='kitti',
        SCORE_THRESH=0.1,
        OUTPUT_RAW_SCORE=False,
        RECALL_THRESH_LIST=[0.3, 0.5, 0.7],
        NMS_CONFIG=dict(NMS_TYPE='nms_gpu',
                        NMS_THRESH=0.01,
                        NMS_PRE_MAXSIZE=4096,
                        NMS_POST_MAXSIZE=500,
                        MULTI_CLASSES_NMS=False, )
    )
)
RUN.tracker.metrics = DATASET.get('metrics', [])
