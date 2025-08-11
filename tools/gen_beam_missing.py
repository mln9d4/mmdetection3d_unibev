import argparse
import os
import random
import time
import glob
import numpy as np
import multiprocessing as mp
import copy
from pathlib import Path
from tqdm import tqdm
from nuscenes import NuScenes
import pickle

seed = 1205
random.seed(seed)
np.random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser(description='LiDAR beam missing')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default=mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data/nuscenes')
    parser.add_argument('-d', '--dst_folder', help='save folder of dataset',
                        default='./save_root/beam_missing/light')  # ['light','moderate','heavy']
    parser.add_argument('-b', '--num_beam_to_drop', help='number of beam to be dropped', type=int, default=8)
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_arguments()
    num_beam_to_drop = args.num_beam_to_drop
    print(num_beam_to_drop)
    print('')
    print(f'using {args.n_cpus} CPUs')

    all_files = []
    nusc_info = NuScenes(version='v1.0-mini', dataroot=args.root_folder, verbose=False)
    imageset = os.path.join(args.root_folder, "mmdet3d_bevformer/mini_nuscenes_infos_temporal_val.pkl")
    with open(imageset, 'rb') as f:
        infos = pickle.load(f)
    all_files = infos['infos']

    all_paths = copy.deepcopy(all_files)
    dst_folder = args.dst_folder

    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    lidar_save_root = os.path.join(dst_folder, 'samples/LIDAR_TOP')
    if not os.path.exists(lidar_save_root):
        os.makedirs(lidar_save_root)
    sweep_save_root = os.path.join(dst_folder, 'sweeps/LIDAR_TOP')
    if not os.path.exists(sweep_save_root):
        os.makedirs(sweep_save_root)

    def reduce_beams(points, beam_id, num_beam_to_drop):
        # drop_range = np.arange(0, 32, 1)
        # Non-random beam dropping
        to_drop = np.linspace(0, 32, num_beam_to_drop, dtype=int)
        for id in to_drop:
            points_to_drop = beam_id == id
            points = np.delete(points, points_to_drop, axis=0)
            beam_id = np.delete(beam_id, points_to_drop, axis=0)
        return points

    def process_file(lidar_path, save_path):
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
        print(f"POINTS {points}")
        beam_id = points[:, -1].astype(np.int64)
        points = reduce_beams(points, beam_id, num_beam_to_drop)
        points.astype(np.float32).tofile(save_path)

    def _map(i: int) -> None:
        info = all_paths[i]
        # Process lidar sample
        lidar_path = info['lidar_path'][16:]
        lidar_full_path = os.path.join(args.root_folder, lidar_path)
        lidar_save_path = os.path.join(dst_folder, lidar_path)
        process_file(lidar_full_path, lidar_save_path)

        # Process lidar sweeps
        if 'sweeps' in info:
            sweeps = info['sweeps']
            for sweep in sweeps:
                sweep_path = sweep['data_path'][16:]
                sweep_full_path = os.path.join(args.root_folder, sweep_path)
                sweep_save_path = os.path.join(dst_folder, sweep_path)
                process_file(sweep_full_path, sweep_save_path)

    n = len(all_files)

    with mp.Pool(args.n_cpus) as pool:
        l = list(tqdm(pool.imap(_map, range(n)), total=n))
