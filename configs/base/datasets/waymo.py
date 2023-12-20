from pathlib import Path
from easydict import EasyDict as dict

DATASET = dict(
    NAME='WaymoDataset',
    TYPE=Path(__file__).stem,
    DATA_PATH='data/waymo',

    POINT_CLOUD_RANGE=[-75.2, -75.2, -2, 75.2, 75.2, 4],
    CLASS_NAMES=['Vehicle', 'Pedestrian', 'Cyclist'],

    DATA_SPLIT=dict(train='train', test='val'),
    PROCESSED_DATA_TAG='waymo_processed_data_v0_5_0',
    SAMPLED_INTERVAL={
        'train': 5,
        'test': 1
    },

    FILTER_EMPTY_BOXES_FOR_TRAIN=True,
    DISABLE_NLZ_FLAG_ON_POINTS=True,
    USE_SHARED_MEMORY=False,
    SHARED_MEMORY_FILE_LIMIT=35000,

    DATA_AUGMENTOR=dict(
        DISABLE_AUG_LIST=['placeholder'],
        AUG_CONFIG_LIST=[
            dict(NAME='gt_sampling',
                 USE_ROAD_PLANE=False,
                 DB_INFO_PATH=['waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl'],
                 DB_DATA_PATH=['waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy'],
                 PREPARE=dict(
                     filter_by_min_points=['Vehicle:5', 'Pedestrian:5', 'Cyclist:5'],
                     filter_by_difficulty=[-1]
                 ),
                 SAMPLE_GROUPS=['Vehicle:15', 'Pedestrian:10', 'Cyclist:10'],
                 NUM_POINT_FEATURES=5,
                 REMOVE_EXTRA_WIDTH=[0.0, 0.0, 0.0],
                 LIMIT_WHOLE_SCENE=True),
            dict(NAME='random_world_flip',
                 ALONG_AXIS_LIST=['x', 'y']),
            dict(NAME='random_world_rotation',
                 WORLD_ROT_ANGLE=[-0.78539816, 0.78539816]),
            dict(NAME='random_world_scaling',
                 WORLD_SCALE_RANGE=[0.95, 1.05])
        ]
    ),
    POINT_FEATURE_ENCODING=dict(
        encoding_type='absolute_coordinates_encoding',
        used_feature_list=['x', 'y', 'z', 'intensity', 'elongation'],
        src_feature_list=['x', 'y', 'z', 'intensity', 'elongation'],
    ),

    DATA_PROCESSOR=[
        dict(NAME='mask_points_and_boxes_outside_range',
             REMOVE_OUTSIDE_BOXES=True),
        dict(NAME='shuffle_points',
             SHUFFLE_ENABLED=dict(train=True, test=True)),
        dict(NAME='transform_points_to_voxels',
             VOXEL_SIZE=[0.1, 0.1, 0.15],
             MAX_POINTS_PER_VOXEL=5,
             MAX_NUMBER_OF_VOXELS=dict(train=150000, test=150000))
    ],
    metrics=[]
)
