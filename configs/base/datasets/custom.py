from pathlib import Path
from easydict import EasyDict as dict

DATASET = dict(
    NAME='CustomDataset',
    TYPE=Path(__file__).stem,
    DATA_PATH='data/custom',
    CLASS_NAMES=['Vehicle', 'Pedestrian', 'Cyclist'],
    GET_ITEM_LIST=["points"],
    POINT_CLOUD_RANGE=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    SWEEPS=dict(
        ENABLED=True,
        RANGE=[-15, 0],
        REMOVE_EGO_RADIUS=1.0,
    ),

    DATA_SPLIT=dict(train='train',
                    test='test'),
    INFO_PATH=dict(train=['infos'],
                   test=['infos']),

    DATA_AUGMENTOR=dict(
        DISABLE_AUG_LIST=['placeholder'],
        AUG_CONFIG_LIST=[
            dict(NAME='gt_sampling',
                 USE_ROAD_PLANE=True,
                 DB_INFO_PATH=['kitti_dbinfos_train.pkl'],
                 PREPARE=dict(filter_by_min_points=['Vehicle:5', 'Pedestrian:5', 'Cyclist:5'],
                              filter_by_difficulty=[-1]),
                 SAMPLE_GROUPS=['Vehicle:20', 'Pedestrian:15', 'Cyclist:15'],
                 NUM_POINT_FEATURES=4,
                 DATABASE_WITH_FAKELIDAR=False,
                 REMOVE_EXTRA_WIDTH=[0.0, 0.0, 0.0],
                 LIMIT_WHOLE_SCENE=True),
            dict(NAME='random_world_flip',
                 ALONG_AXIS_LIST=['x']),
            dict(NAME='random_world_rotation',
                 WORLD_ROT_ANGLE=[-0.78539816, 0.78539816]),
            dict(NAME='random_world_scaling',
                 WORLD_SCALE_RANGE=[0.95, 1.05])
        ]
    ),
    TEST_TIME_AUGMENTOR=dict(
        BOXES_FUSION=dict(
            NAME='nms',
            MATCH=dict(discard=1, radius=1.0),
            BANDWIDTH=dict(bw_loc=1.0, bw_yaw=0.1, bw_score=2.0, bw_label=0.5)
        ),
        TRANSFORMS=[
            dict(name='global_flip', prob=1.0, axis=['x', 'y']),
            dict(name='global_scale', prob=1.0, range=[0.8, 1.2]),
            dict(name='global_rotate', prob=1.0, range=[-3.1415926, 3.1415926]),
        ],
        AUGMENTOR_QUEUE=[[0], [2], [0, 2]],
        ENABLE_INDEX=-1,
    ),
    POINT_FEATURE_ENCODING=dict(
        encoding_type='absolute_coordinates_encoding',
        used_feature_list=['x', 'y', 'z', 'intensity'],
        src_feature_list=['x', 'y', 'z', 'intensity', 'timestamp'],
    ),

    DATA_PROCESSOR=[
        dict(NAME='mask_points_and_boxes_outside_range',
             REMOVE_OUTSIDE_BOXES=True),
        dict(NAME='shuffle_points',
             SHUFFLE_ENABLED={'train': True, 'test': False}),
        dict(NAME='transform_points_to_voxels',
             VOXEL_SIZE=[0.1, 0.1, 0.15],
             MAX_POINTS_PER_VOXEL=5,
             MAX_NUMBER_OF_VOXELS={'train': 80000, 'test': 90000})
    ],
    metrics=[
        dict(key='Vehicle_3d/moderate_R40', summary='best', goal='maximize', save=True),
        dict(key='Pedestrian_3d/moderate_R40', summary='best', goal='maximize'),
        dict(key='Cyclist_3d/moderate_R40', summary='best', goal='maximize')
    ],
    debug=False
)
