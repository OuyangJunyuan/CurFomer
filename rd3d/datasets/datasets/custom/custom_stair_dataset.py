import copy
import pickle
import numpy as np
from easydict import EasyDict

from ..kitti import kitti_utils, kitti_calibration, kitti_object3d
from ..dataset import DATASETS, DatasetTemplate
from ....utils import box_utils, common_utils


@DATASETS.register()
class CustomStairDataset(DatasetTemplate):
    default_class_names = ('StairWay',)
    class_map = {'StairWay': 'StairWay', }
    map_class_to_kitti = {'StairWay': 'Car', }

    def __init__(self, dataset_cfg, training=True, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=dataset_cfg.get('CLASS_NAMES', self.default_class_names),
            training=training, root_path=None, logger=logger,
        )

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_file = self.root_path / 'splits' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_file).readlines()] if split_file.exists() else []

        self.infos = []
        self.seq_infos = {}
        self.include_data(self.mode)

    @staticmethod
    def get_sequence_name(frame_id: str):
        seq_name = frame_id.split(sep='_')
        return '_'.join(seq_name[:-1]), seq_name[-1]

    @staticmethod
    def get_frame_id(seq, idx):
        return '%s_%s' % (seq, idx)

    def get_label(self, label_file):
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, seq, idx):
        lidar_file = self.root_path / self.seq_infos[seq][idx]['point_cloud']['data_path']
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_pose(self, seq, idx):
        return self.seq_infos[seq][idx]['pose']

    def get_timestamp(self, seq, idx):
        return self.seq_infos[seq][idx]['timestamp'] * 1e-9

    def get_lidar_with_sweeps(self, seq, idx, offset):
        """
        data padding when no enough frames ahead this frame.
        WOD: first element padding
        NuScenes: last element padding
        """

        def remove_ego_points(points, center_radius):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        pose_cur = self.get_pose(seq, idx)
        timestamp_cur = self.get_timestamp(seq, idx)
        points_list = []
        start, end, step = (idx + offset[1], idx + offset[0] - 1, -1 * (offset[2] if len(offset) > 2 else 1))
        sweep_indices = np.clip(list(range(start, end, step)), 0, 0x7FFFFFFF)
        assert len(sweep_indices) == (offset[1] - offset[0] + 1) // (-step)

        for sweep_idx in sweep_indices:
            pose = self.get_pose(seq, sweep_idx)
            points = self.get_lidar(seq, sweep_idx)
            timestamp = self.get_timestamp(seq, sweep_idx)
            motion = np.linalg.inv(pose_cur) @ pose

            points = remove_ego_points(points, self.dataset_cfg.SWEEPS.REMOVE_EGO_RADIUS)
            points[:, :3] = (np.hstack((points[:, :3], np.ones((points.shape[0], 1)))) @ motion.T)[:, :3]
            points = np.hstack((points, (timestamp_cur - timestamp) * np.ones_like(points[:, :1])))

            points_list.append(points)

        points = np.concatenate(points_list, axis=0)
        return points

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_file = self.root_path / 'splits' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_file).readlines()] if split_file.exists() else []

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')

        for info_dir in self.dataset_cfg.INFO_PATH[mode]:
            for sequence in self.sample_sequence_list:
                info_path = self.root_path / info_dir / f'{sequence}.pkl'
                if not info_path.exists():
                    continue

                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    self.infos.extend(infos)
                    self.seq_infos[sequence] = infos
            self.logger.info('Total samples for CUSTOM dataset: %d' % (len(self.infos)))

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        info = EasyDict(copy.deepcopy(self.infos[index]))
        input_dict = {
            'index': index,
            'frame_id': info.frame_id
        }

        if 'points' in self.dataset_cfg.GET_ITEM_LIST:
            if self.dataset_cfg.SWEEPS.ENABLED:
                points = self.get_lidar_with_sweeps(info.sequence, info.sample_idx, self.dataset_cfg.SWEEPS.RANGE)
            else:
                points = self.get_lidar(info.sequence, info.sample_idx)
            input_dict['points'] = points

        if 'image' in self.dataset_cfg.GET_ITEM_LIST:
            pass

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    # def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
    #     import concurrent.futures as futures
    #
    #     def process_single_scene(sample_idx):
    #         print('%s sample_idx: %s' % (self.split, sample_idx))
    #         info = {}
    #         pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
    #         info['point_cloud'] = pc_info
    #
    #         if has_label:
    #             annotations = {}
    #             gt_boxes_lidar, name = self.get_label(sample_idx)
    #             annotations['name'] = name
    #             annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
    #             info['annos'] = annotations
    #
    #         return info
    #
    #     sample_id_list = sample_id_list if sample_id_list is not None else self.sample_sequence_list
    #
    #     # create a thread pool to improve the velocity
    #     with futures.ThreadPoolExecutor(num_workers) as executor:
    #         infos = executor.map(process_single_scene, sample_id_list)
    #     return list(infos)
    #
    # def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
    #     import torch
    #
    #     database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
    #     db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)
    #
    #     database_save_path.mkdir(parents=True, exist_ok=True)
    #     all_db_infos = {}
    #
    #     with open(info_path, 'rb') as f:
    #         infos = pickle.load(f)
    #
    #     for k in range(len(infos)):
    #         print('gt_database sample: %d/%d' % (k + 1, len(infos)))
    #         info = infos[k]
    #         sample_idx = info['point_cloud']['lidar_idx']
    #         points = self.get_lidar(sample_idx)
    #         annos = info['annos']
    #         names = annos['name']
    #         gt_boxes = annos['gt_boxes_lidar']
    #
    #         num_obj = gt_boxes.shape[0]
    #         point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
    #             torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
    #         ).numpy()  # (nboxes, npoints)
    #
    #         for i in range(num_obj):
    #             filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
    #             filepath = database_save_path / filename
    #             gt_points = points[point_indices[i] > 0]
    #
    #             gt_points[:, :3] -= gt_boxes[i, :3]
    #             with open(filepath, 'w') as f:
    #                 gt_points.tofile(f)
    #
    #             if (used_classes is None) or names[i] in used_classes:
    #                 db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
    #                 db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
    #                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
    #                 if names[i] in all_db_infos:
    #                     all_db_infos[names[i]].append(db_info)
    #                 else:
    #                     all_db_infos[names[i]] = [db_info]
    #
    #     # Output the num of all classes in database
    #     for k, v in all_db_infos.items():
    #         print('Database %s: %d' % (k, len(v)))
    #
    #     with open(db_info_save_path, 'wb') as f:
    #         pickle.dump(all_db_infos, f)
    #
    # @staticmethod
    # def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
    #     with open(save_label_path, 'w') as f:
    #         for idx in range(gt_boxes.shape[0]):
    #             boxes = gt_boxes[idx]
    #             name = gt_names[idx]
    #             if name not in class_names:
    #                 continue
    #             line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
    #                 x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
    #                 w=boxes[4], h=boxes[5], angle=boxes[6], name=name
    #             )
    #             f.write(line)


def create_custom_infos(dataset_cfg, workers=4):
    def run_lidar_odom(sequence_path):
        import os
        try:
            import kiss_icp
        except ImportError as e:
            os.system("pip install kiss-icp")

        _ = os.system(f"cd {str(sequence_path)} && kiss_icp_pipeline pointclouds")
        result_path = sequence_path / 'results'
        odom_path = result_path / 'latest/pointclouds_poses.npy'
        save_path = sequence_path / 'pointclouds_poses.npy'
        os.system(f"mv {str(odom_path)} {str(save_path)}")
        os.system(f"rm -rf {str(result_path)}")

    def simulate_timestamps(num_samples, hz=10):
        with open(timestamps_path, 'w') as f:
            fake_times = ['%023d' % int(1 / hz * i * 10 ** 9) for i in range(num_samples)]
            f.write('\n'.join(fake_times))

    dataset = CustomStairDataset(dataset_cfg, training=False)
    sequences_root = dataset.root_path / "sequences"
    infos_root = dataset.root_path / 'infos'

    print('---------------Start to generate data infos---------------')
    from tqdm import tqdm

    infos_root.mkdir(exist_ok=True, parents=True)
    for sequence in tqdm(iterable=sequences_root.iterdir(), desc='sequences', leave=True):
        infos = []
        poses_path = sequence / 'pointclouds_poses.npy'
        timestamps_path = sequence / 'timestamps.txt'
        points_paths = sorted(list((sequence / 'pointclouds').iterdir()))
        labels_paths = sorted((list((sequence / 'labels').iterdir())))

        if not poses_path.exists():
            run_lidar_odom(sequence)
        poses = np.load(poses_path)

        if not timestamps_path.exists():
            simulate_timestamps(len(points_paths))
        timestamps = open(timestamps_path, 'r').readlines()

        labels = [dataset.get_label(path) for path in labels_paths]

        for sample_idx, (points_path, timestamp, label, pose) in enumerate(
                zip(points_paths, timestamps, labels, poses)
        ):
            pc_info = {
                'data_path': points_path.relative_to(dataset.root_path).__str__(),
                'num_features': 4
            }
            image_info = dict(
                # sample_idx=sample_idx,
                # image_shape=dataset.get_image_shape(sample_idx)
            )
            # calib = dataset.get_calib_seq(frame_dict['point_cloud']['lidar_sequence'])
            calib_info = dict(
                # P2=None,
                # RO_rect=calib.R0,
                # Tr_velo_to_cam=calib.V2C
            )
            # frame_dict['calib'] = {'P2': calib.P2, 'R0_rect': calib.R0, 'Tr_velo_to_cam': calib.V2C}

            annotations = {'gt_boxes_lidar': label[0], 'name': label[1]}
            # kitti_utils.transform_annotations_to_kitti_format([annotations], map_name_to_kitti=CLASS_MAPPING)

            info = {
                'frame_id': dataset.get_frame_id(sequence.name, points_path.stem),
                'sequence': sequence.name,
                'sample_idx': sample_idx,
                'timestamp': int(timestamp),

                'pose': pose,
                'point_cloud': pc_info,
                'image': image_info,
                'calib': calib_info,
                'annos': annotations
            }
            infos.append(info)
        with open(f'{infos_root / sequence.name}.pkl', 'wb') as f:
            pickle.dump(infos, f)
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse
    from ....core.config import Config, PROJECT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/base/datasets/custom.py')
    parser.add_argument('--name', type=str, default='custom')
    args = parser.parse_args()

    create_infos_dataset_cfg = Config.fromfile_py(args.cfg).DATASET
    # create_infos_dataset_cfg.DATA_PATH = PROJECT_ROOT / 'data' / args.name
    create_custom_infos(dataset_cfg=create_infos_dataset_cfg)
