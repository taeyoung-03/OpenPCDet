"""
once_dataset_no_cam.py

A version of ONCEDataset with all camera-related logic (cam_names, cam_tags,
calib, boxes_2d, point_painting, image loading, etc.) removed from the
original ONCEDataset code, restructured Kriso-style to work purely on
"LiDAR points + 3D annotations".

Summary of changes vs. the original ONCEDataset
-------------------------------------------------
1. Removed the cam_names / cam_tags attributes (cameras are not used at all)
2. Still uses the Octopus toolkit, but only load_point_cloud
   (removed calls to camera-related methods like load_image,
   project_lidar_to_image, etc.)
3. Removed the entire point_painting() method (it requires segmentation
   images from cameras, which is fundamentally incompatible with a
   camera-free pipeline)
4. In get_infos(), removed the per-frame code that read cam01~cam09 image
   paths / calibration (calib_dict, frame_dict[cam_name], etc.)
5. Removed boxes_2d (camera-based 2D boxes) from annos_dict, keeping only
   boxes_3d / name
6. Changed the execution style of get_infos() to as_completed-based, like
   the Kriso version (the original used executor.map, which preserves
   order -- switch back to executor.map if ordering matters)
7. Kept Patch 1~4 from the original ONCE code (guard for missing splits,
   splits as an argument, etc.) unchanged

Note: once_toolkits.py (the Octopus class) does NOT need any changes.
Its __init__ only reads the 'calib' field from each sequence's JSON
(plain metadata), and only load_image() / project_lidar_to_image() touch
actual camera jpg files on disk -- neither of which this class calls.
"""

import copy
import pickle

import numpy as np
import torch
from pathlib import Path

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils
from .once_toolkits import Octopus


class ONCEDatasetNoCam(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        ONCE dataset variant that does not use any camera information.

        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']
        assert self.split in ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        # Camera-related attributes (cam_names, cam_tags) fully removed.
        # The Octopus toolkit is used only for LiDAR loading.
        self.toolkits = Octopus(self.root_path)

        self.once_infos = []
        self.include_once_data(self.split)

    def include_once_data(self, split):
        if self.logger is not None:
            self.logger.info('Loading ONCE dataset (no-camera mode)')
        once_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                once_infos.extend(infos)

        def check_annos(info):
            return 'annos' in info

        if self.split != 'raw':
            once_infos = list(filter(check_annos, once_infos))

        self.once_infos.extend(once_infos)

        if self.logger is not None:
            self.logger.info('Total samples for ONCE dataset: %d' % (len(once_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, sequence_id, frame_id):
        return self.toolkits.load_point_cloud(sequence_id, frame_id)

    # get_image / project_lidar_to_image / point_painting and other
    # camera-related methods have all been removed.

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.once_infos) * self.total_epochs
        return len(self.once_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)

        # Removed the POINT_PAINTING option branch (needs camera segmentation,
        # not supported here)

        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    def get_infos(self, num_workers=4, sample_seq_list=None):
        import concurrent.futures as futures
        import json
        root_path = self.root_path

        """
        # Original dataset json format (for reference only, camera fields are ignored)
        {
            'meta_info':
            'calib': {...}          # <- ignored
            'frames': [
                {
                    'frame_id': timestamp,
                    'annos': {
                        'names': list
                        'boxes_3d': list of list
                        'boxes_2d': {...}   # <- ignored
                    }
                    'pose': list
                },
                ...
            ]
        }

        # The open pcdet format produced by this version (no camera fields)
        {
            'meta_info':
            'sequence_id': seq_idx
            'frame_id': timestamp
            'timestamp': timestamp
            'lidar': path
            'pose': np.array
            'annos': {
                'name': np.array
                'boxes_3d': np.array
                'num_points_in_gt': np.array
            }
        }
        """

        def process_single_sequence(seq_idx):
            print('%s seq_idx: %s' % (self.split, seq_idx))
            seq_infos = []
            seq_path = Path(root_path) / 'data' / seq_idx
            json_path = seq_path / ('%s.json' % seq_idx)
            with open(json_path, 'r') as f:
                info_this_seq = json.load(f)
            meta_info = info_this_seq.get('meta_info', '')
            # calib info is not read here (cameras are not used)

            for f_idx, frame in enumerate(info_this_seq['frames']):
                frame_id = frame['frame_id']
                prev_id = info_this_seq['frames'][f_idx - 1]['frame_id'] if f_idx > 0 else None
                next_id = (info_this_seq['frames'][f_idx + 1]['frame_id']
                           if f_idx < len(info_this_seq['frames']) - 1 else None)
                pc_path = str(seq_path / 'lidar_roof' / ('%s.bin' % frame_id))
                pose = np.array(frame.get('pose', np.eye(4).tolist()))

                frame_dict = {
                    'sequence_id': seq_idx,
                    'frame_id': frame_id,
                    'timestamp': int(frame_id),
                    'prev_id': prev_id,
                    'next_id': next_id,
                    'meta_info': meta_info,
                    'lidar': pc_path,
                    'pose': pose
                    # No camera image path / calib_dict fields
                }

                if 'annos' in frame:
                    annos = frame['annos']
                    boxes_3d = np.array(annos['boxes_3d'])
                    if boxes_3d.shape[0] == 0:
                        print(frame_id)
                        continue

                    annos_dict = {
                        'name': np.array(annos['names']),
                        'boxes_3d': boxes_3d,
                        # No boxes_2d field (cameras are not used)
                    }

                    points = self.get_lidar(seq_idx, frame_id)
                    corners_lidar = box_utils.boxes_to_corners_3d(np.array(annos['boxes_3d']))
                    num_gt = boxes_3d.shape[0]
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annos_dict['num_points_in_gt'] = num_points_in_gt

                    frame_dict.update({'annos': annos_dict})

                seq_infos.append(frame_dict)
            return seq_infos

        # Guard against empty split (same as original Patch 1)
        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        if not sample_seq_list:
            return []

        # Kriso-style: as_completed based (order doesn't matter, collected as they finish)
        all_infos = []
        with futures.ThreadPoolExecutor(num_workers) as executor:
            future_to_seq = {
                executor.submit(process_single_sequence, seq): seq
                for seq in sample_seq_list
            }
            for future in futures.as_completed(future_to_seq):
                seq_infos = future.result()
                all_infos.extend(seq_infos)

        return all_infos

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('once_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            if 'annos' not in infos[k]:
                continue
            print('gt_database sample: %d' % (k + 1))
            info = infos[k]
            frame_id = info['frame_id']
            seq_id = info['sequence_id']
            points = self.get_lidar(seq_id, frame_id)

            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['boxes_3d']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))
                db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                           'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            return {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
            }

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from .once_eval.evaluation import get_evaluation_results

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.once_infos]
        ap_result_str, ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict


def create_once_infos_no_cam(dataset_cfg, class_names, data_path, save_path, workers=4, splits=None):
    dataset = ONCEDatasetNoCam(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)

    if splits is None:
        splits = ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large']
    ignore = ['test']

    print('---------------Start to generate data infos (no-camera)---------------')

    for split in splits:
        if split in ignore:
            continue

        dataset.set_split(split)

        if not dataset.sample_seq_list:
            print('[skip] ImageSets/%s.txt not found, skipping info generation for split "%s".' % (split, split))
            continue

        filename = 'once_infos_%s.pkl' % split
        filename = save_path / Path(filename)
        once_infos = dataset.get_infos(num_workers=workers)
        with open(filename, 'wb') as f:
            pickle.dump(once_infos, f)
        print('ONCE info %s file is saved to %s' % (split, filename))

    if 'train' in splits:
        train_filename = save_path / 'once_infos_train.pkl'
        if train_filename.exists():
            print('---------------Start create groundtruth database for data augmentation---------------')
            dataset.set_split('train')
            dataset.create_groundtruth_database(train_filename, split='train')

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_once_infos_no_cam', help='')
    parser.add_argument('--runs_on', type=str, default='server', help='')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                         help='List of splits to generate (e.g. --splits train). '
                              'If not given, all 6 splits are attempted.')
    args = parser.parse_args()

    if args.func == 'create_once_infos_no_cam':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        once_data_path = ROOT_DIR / 'data' / 'once'
        once_save_path = ROOT_DIR / 'data' / 'once'

        if args.runs_on == 'cloud':
            once_data_path = Path('/cache/once/')
            once_save_path = Path('/cache/once/')
            dataset_cfg.DATA_PATH = dataset_cfg.CLOUD_DATA_PATH

        create_once_infos_no_cam(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Bus', 'Truck', 'Pedestrian', 'Bicycle'],
            data_path=once_data_path,
            save_path=once_save_path,
            splits=args.splits
        )
