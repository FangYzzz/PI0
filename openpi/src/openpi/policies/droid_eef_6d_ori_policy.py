import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/eef_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Droid_EEF_Inputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        
        #----------------------------quat -> 6d rot----------------------------#
        state_trans= data["observation/eef_position"][...,:3]
        state_quat = data["observation/eef_position"][...,3:7]
        state_rot = R.from_quat(state_quat).as_matrix()[..., :, :2]
        # print("state_quat",state_quat.shape)
        # print("state_trans",state_trans.shape)
        if len(state_rot.shape)==3:
            B = state_rot.shape[0]
            rot6 = state_rot[..., :, :2]
            state_rot6d = np.concatenate([rot6[...,:,0],rot6[...,:,1]], axis=-1)
        else:
            rot6 = state_rot[..., :, :2]
            state_rot6d = np.concatenate([rot6[...,:,0],rot6[...,:,1]], axis=-1)
        state = np.concatenate([state_trans, state_rot6d, gripper_pos], axis=-1)
        # print(state_q)
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            action_trans= data["actions"][...,:3]
            action_quat = data["actions"][...,3:7]
            action_rot = R.from_quat(action_quat).as_matrix()
            # print("action_quat",action_quat.shape)
            # print("action_trans",action_trans.shape)
            if len(action_rot.shape)==3:
                B = action_rot.shape[0]
                rot6 = action_rot[..., :, :2]
                # action_rot6d = rot6.reshape(B, 6)
                action_rot6d = np.concatenate([rot6[...,:,0],rot6[...,:,1]], axis=-1)
            else:
                rot6 = action_rot[..., :, :2]
                action_rot6d = np.concatenate([rot6[...,:,0],rot6[...,:,1]], axis=-1)
            action = np.concatenate([action_trans, action_rot6d, data["actions"][...,-1][...,None]], axis=-1)
            inputs["actions"] = action

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Droid_EEF_Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :10])}
