import pyzed.sl as sl
import cv2
import numpy as np

from loguru import logger
import os
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper

class Camera_Read:
    def __init__(self,camera_kwargs={}):
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
    def prepare_image_256(self, img, size=(256, 256)):
        """中心裁剪成正方形，再缩放到指定大小 (默认 256x256)，输出 RGB uint8。"""
        if img is None:
            return None
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img[y0:y0 + side, x0:x0 + side]
        out = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        return out.astype(np.uint8, copy=False)

    def get_images(self, args, camera_kwargs={}):
        camera_obs, camera_timestamp = self.camera_reader.read_cameras()

        image_observations = camera_obs['image']
        left_image, right_image, wrist_image = None, None, None
        for key in image_observations:
            # Note the "left" below refers to the left camera in the stereo pair.
            # The model is only trained on left stereo cams, so we only feed those.
            if args.left_camera_id in key and "left" in key:
                left_image = image_observations[key]
            # elif args.right_camera_id in key and "left" in key:
            #     right_image = image_observations[key]
            elif args.wrist_camera_id in key and "left" in key:
                wrist_image = image_observations[key]

        # Drop the alpha dimension
        left_image = left_image[..., :3]
        # right_image = right_image[..., :3]
        wrist_image = wrist_image[..., :3]

        # Convert to RGB
        left_image = left_image[..., ::-1]
        # right_image = right_image[..., ::-1]
        wrist_image = wrist_image[..., ::-1]

        left_image = self.prepare_image_256(left_image)
        wrist_image = self.prepare_image_256(wrist_image)

        return {
            "left_image": left_image,
            # "right_image": right_image,
            "wrist_image": wrist_image,
        }

