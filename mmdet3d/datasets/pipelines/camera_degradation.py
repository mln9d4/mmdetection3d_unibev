import mmcv
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES

"""
Custom module built by MD Yang to degrade camera sensor
using various tactics.
"""

@PIPELINES.register_module()
class DownscaleUpscale(object):
    """Simulate downgraded camera by down and upscaling the camera image,
    simulating a lower resolution by losing information.
    
    You can specify a resolution you want to scale it to.

    """
    def __init__(self,
                target_width=1280,
                target_height=720,
                original_width=1600,
                original_height=900,
                save_location="/home/mingdayang/mmdetection3d/figures/", 
                save_folder_name="camera_degradation",
                save_fig=False):
        self.target_width=target_width
        self.target_height=target_height
        self.original_width=original_width
        self.original_height=original_height
        self.save_location=save_location
        self.save_folder_name=save_folder_name
        self.save_fig=save_fig


    def _down(self, image):
        fx = self.target_width/self.original_width
        fy = self.target_height/self.original_height
        # return cv2.resize(image, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        # (width, height)
        down = cv2.resize(image, (self.target_width, self.target_height))
        return down

    def _up(self, image):
        fx = self.original_width/self.target_width
        fy = self.original_height/self.target_height
        # up_scaled = cv2.resize(image, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        up_scaled = cv2.resize(image, (self.original_width, self.original_height))
        return up_scaled

    def _reduce(self, image):
        down_scaled = self._down(image)
        up_scaled = self._up(down_scaled)
        return up_scaled

    def _visualize_images(self, image_original, image_modified, sample_idx, i):

        # fig, axes = plt.subplots(1, 2, figsize=(12,6))
        # Making sure rgb values are between 0, 255 and in integers
        image_original = image_original.astype(np.uint8)
        image_modified = image_modified.astype(np.uint8)

        image_original_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        image_modified_rgb = cv2.cvtColor(image_modified, cv2.COLOR_BGR2RGB)

        # axes[0].imshow(image_original_rgb)
        # axes[0].set_title(f"Original image {self.original_width}x{self.original_height} {i}")

        # axes[1].imshow(image_modified_rgb)
        # axes[1].set_title(f"Downscaled upscaled image {self.target_width}x{self.target_height} {i}")

        # # Adjust spacing between plots
        # plt.tight_layout()

        # Show the plots
        # sub_folder = 'mini_nuscenes'
        # folder_path = os.path.join(self.save_location, self.save_folder_name)
        # folder_path = os.path.join(folder_path, sub_folder)
        # os.makedirs(folder_path, exist_ok=True) 

        # file_path = os.path.join(folder_path, f"camera_degredation_{self.target_width}x{self.target_height}_{sample_idx}_{i}.jpg")
        # plt.savefig(file_path)
        # plt.close(fig)

        sub_folder = 'mini_nuscenes'
        org_folder = 'original'
        res_folder = f'modified_{self.target_width}x{self.target_height}'
        folder_path = os.path.join(self.save_location, self.save_folder_name)
        folder_path = os.path.join(folder_path, sub_folder)
        folder_res_path = os.path.join(folder_path, res_folder)
        folder_or_path = os.path.join(folder_path, org_folder)
        os.makedirs(folder_res_path, exist_ok=True) 
        os.makedirs(folder_or_path, exist_ok=True)

        res_image = os.path.join(folder_res_path, f"camera_degredation_{self.target_width}x{self.target_height}_{sample_idx}_{i}.jpg")
        or_image = os.path.join(folder_or_path, f"camera_degradation_original_{sample_idx}_{i}.jpg")

        cv2.imwrite(or_image, image_original)
        cv2.imwrite(res_image, image_modified)

    def __call__(self, results):
        image = results['img'][0]
        self.original_width = np.shape(image)[1]
        self.original_height = np.shape(image)[0]

        for i, image in enumerate(results['img']):
            # print(results.keys())
            modified_image = self._reduce(image)
            results['img'][i] = modified_image
            if self.save_fig:
                # test = self._down(image)
                self._visualize_images(image, modified_image, results['sample_idx'], i)
            
        return results