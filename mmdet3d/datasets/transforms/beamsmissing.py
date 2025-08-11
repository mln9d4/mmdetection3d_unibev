import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES

"""
Custom module built by MD Yang to remove beams from Lidar data of Nuscenes dataset.
"""


@PIPELINES.register_module()
class RemoveLiDARBeamsSpaced(object):
    """Remove points from 3D LiDAR point cloud beam-wise. The beam spacing is as
    close as possible to being even to simualate a down-graded sensor.

    Args:

    
    """
    def __init__(self, 
                num_beam_to_drop=8, 
                num_beam_sensor=32, 
                coord_type='LIDAR', 
                save_fig=False, 
                save_location="/home/mingdayang/mmdetection3d/figures/", 
                save_folder_name="experiment2"):
        self.num_beam_to_drop = num_beam_to_drop
        self.num_beam_sensor = num_beam_sensor
        self.coord_type = coord_type
        self.save_location = save_location
        self.save_folder_name = save_folder_name
        self.save_fig = save_fig
    
    def _reduce_beams(self, points):
        beam_id = points[:, -1].astype(np.int64)

        to_drop = np.linspace(0, self.num_beam_sensor, self.num_beam_to_drop, dtype=int)
        for id in to_drop:
            points_to_drop = beam_id == id
            points = np.delete(points, points_to_drop, axis=0)
            beam_id = np.delete(beam_id, points_to_drop, axis=0)

        # print(f"After dropping: {np.unique(points[:, -1], return_counts=True)}")
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        return points

    def _visualize_points(self, points, original_points, sample_idx):
        """Save image of points"""

        fig, axes = plt.subplots(1, 2, figsize=(12,6))\
        
        # Plot the first set of LiDAR points
        axes[0].scatter(original_points[:, 0], original_points[:, 1], s=1, c='blue')
        axes[0].set_title("LiDAR BEV Plot original")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_xlim([-50, 50])
        axes[0].set_ylim([-50, 50])
        axes[0].grid(True)

        # Plot the second set of LiDAR points
        axes[1].scatter(points[:, 0], points[:, 1], s=1, c='red')
        axes[1].set_title(f"LiDAR BEV Plot removing {self.num_beam_to_drop}")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[1].set_xlim([-50, 50])
        axes[1].set_ylim([-50, 50])
        axes[1].grid(True)

        # Adjust spacing between plots
        plt.tight_layout()

        # Show the plots
        sub_folder = 'samples'
        folder_path = os.path.join(self.save_location, self.save_folder_name)
        folder_path = os.path.join(folder_path, sub_folder)
        os.makedirs(folder_path, exist_ok=True) 

        file_path = os.path.join(folder_path, f"L_beams_{self.num_beam_to_drop}_dropped_sample_{sample_idx}.jpg")
        plt.savefig(file_path)
        plt.close(fig)

    def __call__(self, results):
        """Call fuctions to load points and reduce the beams
        Args:
            results (dict): results dict from :obj:'mmdet.CustomDataset'
        
        Returns:
            dict: The dict contains the reduced lidar point cloud.
        """
        # print(f"Results dict keys: {results.keys()}")
        # print(f"Points, counts -1: {np.unique(results['points'][:,-1], return_counts=True)}")
        original_points = results['points'].tensor.numpy()
        points = results['points'].tensor.numpy()  # Convert points to numpy
        points = self._reduce_beams(points)
        results['points'] = points  # Set back the reduced points
        if self.save_fig:
            self._visualize_points(points, original_points, results['sample_idx'])

        return results 

@PIPELINES.register_module()
class LoadPointsFromMultiSweepsReducedBeams(object):
    """Load points from multiple sweeps

    This is useally used for nuScenes dataset to utilize previous sweeps.

    Modified version to correctly load in the sweeps with reduced amount of beams.

    """
    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 num_beam_to_drop=8,
                 num_beam_sensor=32,
                 coord_type='LIDAR',
                 save_location="/home/mingdayang/mmdetection3d/figures/",
                 save_folder_name='experiment2',
                 save_fig=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.num_beam_to_drop = num_beam_to_drop
        self.num_beam_sensor = num_beam_sensor
        self.coord_type = coord_type
        self.save_location = save_location
        self.save_folder_name = save_folder_name
        self.save_fig = save_fig

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        points = points.reshape(-1, self.load_dim)
        return points
    
    def _reduce_beams(self, points):
        beam_id = points[:, -1].astype(np.int64)

        to_drop = np.linspace(0, self.num_beam_sensor, self.num_beam_to_drop, dtype=int)
        for id in to_drop:
            points_to_drop = beam_id == id
            points = np.delete(points, points_to_drop, axis=0)
            beam_id = np.delete(beam_id, points_to_drop, axis=0)

        # points_class = get_points_type(self.coord_type)
        # points = points_class(
        #     points, points_dim=points.shape[-1], attribute_dims=None
        # )
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def _visualize_points(self, points, original_points, sample_idx):
        """Save image of points"""

        fig, axes = plt.subplots(1, 2, figsize=(12,6))\
        
        # Plot the first set of LiDAR points
        axes[0].scatter(original_points[:, 0], original_points[:, 1], s=1, c='blue')
        axes[0].set_title("LiDAR BEV Plot original")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_xlim([-50, 50])
        axes[0].set_ylim([-50, 50])
        axes[0].grid(True)

        # Plot the second set of LiDAR points
        axes[1].scatter(points[:, 0], points[:, 1], s=1, c='red')
        axes[1].set_title(f"LiDAR BEV Plot removing {self.num_beam_to_drop}")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[1].set_xlim([-50, 50])
        axes[1].set_ylim([-50, 50])
        axes[1].grid(True)

        # Adjust spacing between plots
        plt.tight_layout()

        # Show the plots
        sub_folder = 'sweep'
        folder_path = os.path.join(self.save_location, self.save_folder_name)
        folder_path = os.path.join(folder_path, sub_folder)
        os.makedirs(folder_path, exist_ok=True) 

        file_path = os.path.join(folder_path, f"L_beams_{self.num_beam_to_drop}_dropped_sweep_{sample_idx}.jpg")
        plt.savefig(file_path)
    
    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        original_points = results['points']
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = self._reduce_beams(points_sweep)
                points_sweep = np.copy(points_sweep)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        if self.save_fig:
            self._visualize_points(points, original_points, results['sample_idx'])
        return results
