from ..base.datasets.waymo import *
from ..base.runtime.adam_onecycle_2_20e import *

RUN.max_epochs = 30
RUN.workflows.train = [dict(state='train', split='train', epochs=20),
                       dict(state='test', split='test', epochs=1),
                       dict(state='train', split='train', epochs=5)]

DATASET.SAMPLED_INTERVAL = {'train': 5, 'test': 5}
DATASET.POINT_FEATURE_ENCODING = dict(
    encoding_type='absolute_coordinates_encoding',
    used_feature_list=['x', 'y', 'z', 'intensity'],
    src_feature_list=['x', 'y', 'z', 'intensity', 'elongation'],
)
DATASET.DATA_PROCESSOR[2] = dict(NAME='sample_points_by_voxel',
                                 VOXEL_SIZE=[0.1, 0.1, 0.15],
                                 NUMBER_OF_VOXELS=dict(train=16384 * 4, test=16384 * 4))

MODEL = dict(
    NAME='PointVote',
    CLASS_NAMES=DATASET.CLASS_NAMES,
    BACKBONE_3D=dict(
        NAME='GeneralPointNet2MSG',
        ENCODER=
        [
            # voxel 65535 -> 16384: [0.50296668 0.50296668 0.75445002]
            # voxel 16384 -> 4096: [1.4210464 1.4210464 2.1315696]
            dict(samplers=[dict(name='havs', sample=16384, voxel=[0.5, 0.5, 0.75])],
                 groupers=[dict(name='ball', query=dict(radius=0.2, neighbour=16), mlps=[16, 16, 32]),
                           dict(name='ball', query=dict(radius=0.8, neighbour=32), mlps=[32, 32, 64])],
                 aggregation=dict(name='cat-mlps', mlps=[64])),
            dict(samplers=[dict(name='havs', sample=4096, voxel=[1.4, 1.4, 2.1])],
                 groupers=[dict(name='ball', query=dict(radius=0.8, neighbour=16), mlps=[64, 64, 128]),
                           dict(name='ball', query=dict(radius=1.6, neighbour=32), mlps=[64, 96, 128])],
                 aggregation=dict(name='cat-mlps', mlps=[128])),
            dict(samplers=[dict(name='ctr', sample=2048, mlps=[128], class_names=DATASET.CLASS_NAMES,
                                train=dict(target={'extra_width': [0.5, 0.5, 0.5]},
                                           loss={'weight': 1.0, 'tb_tag': 'sasa_1'}))],
                 groupers=[dict(name='ball', query=dict(radius=1.6, neighbour=16), mlps=[128, 128, 256]),
                           dict(name='ball', query=dict(radius=4.8, neighbour=32), mlps=[128, 256, 256])],
                 aggregation=dict(name='cat-mlps', mlps=[256])),
        ]),
    POINT_HEAD=dict(
        NAME='PointHeadVotePlus',
        CLASS_AGNOSTIC=False,
        VOTE_SAMPLER=dict(name='ctr', sample=1024, mlps=[256], class_names=DATASET.CLASS_NAMES,
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
                       use_mean_size=True, mean_size=[[4.7, 2.1, 1.7],
                                                      [0.91, 0.86, 1.73],
                                                      [1.78, 0.84, 1.78]]),
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
                        NMS_THRESH=0.2,
                        NMS_PRE_MAXSIZE=4096,
                        NMS_POST_MAXSIZE=500,
                        MULTI_CLASSES_NMS=False, )
    )
)
