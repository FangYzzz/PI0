import argparse
import os
from franky import Affine
import franky
import numpy as np
from franky import *
from scipy.spatial.transform import Rotation
import time
import cv2

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Teleoperation for Mobile Bimanual Manipulator Isaac Lab environments."
)
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)

import torch
from pathlib import Path
from oculus.oculus_v0 import Oculus_mobile
from image_read import Camera_Read
from typing import Optional
import shutil
from PIL import Image

HF_LEROBOT_HOME = Path("/home/yuan/VLA/tele_op/lerobot")
REPO_NAME = "lerobot_velocity_chunk_joint" # (with pause)  # lerobot_pickup_pepper_absolute_joint(without pause)
os.environ["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


IMAGE_SHAPE = (224, 224, 3)   


def resize_with_pad_pil(image: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    image = Image.fromarray(image)
    
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    zero_image = np.array(zero_image)
    return zero_image

def connect_to_franka():
    robot = Robot("9.9.9.9") 
    robot.relative_dynamics_factor = 0.05

    return robot

def connect_to_gripper():
    gripper = franky.Gripper("9.9.9.9")

    return gripper

def grasp(gripper):
    speed = 0.02  # 0.02[m/s]
    force = 20.0  # 20.0[N]
    success = gripper.grasp(0.0, speed, force, epsilon_outer=1.0)
    # Move the fingers to a specific width (5cm)
    # success = gripper.move(0.05, speed)

    # Wait for 5s
    # success_future = gripper.move_async(0.05, speed)
    # success_future.wait(5)

def release_gripper(gripper):
    speed = 0.02  # 0.02[m/s]
    gripper.open(speed)

def move_to_home(robot):
    robot.relative_dynamics_factor = 0.1
    # q_start = np.array([0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398])
    q_start = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
    motion = JointMotion(q_start, ReferenceType.Absolute)
    robot.move(motion)
    robot.relative_dynamics_factor = 0.05

class Args:
    # Hardware parameters
    left_camera_id: str = "24285872" # 26658469
    right_camera_id: str = None # "<your_camera_id>"
    wrist_camera_id: str = "11022812"  # "11022812"

    # Policy parameters
    external_camera: Optional[str] = (
        # None  # which external camera should be fed to the policy, choose from ["left", "right"]
        "left"
    )

    # Rollout parameters
    max_timesteps: int = 600 # 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


def main():
    output_path = HF_LEROBOT_HOME / REPO_NAME
    # output_path.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        dataset = LeRobotDataset(repo_id=REPO_NAME)
        # Start a fresh buffer for the next episode index (meta.total_episodes)
        dataset.episode_buffer = dataset.create_episode_buffer()
        # shutil.rmtree(output_path)
    else:    
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="panda",
            fps=15,  # DROID data is typically recorded at 15fps
            features={
                # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
                "exterior_image_1_left": {
                    "dtype": "image",
                    "shape": IMAGE_SHAPE,  # This is the resolution used in the DROID RLDS dataset (180, 320, 3)
                    "names": ["height", "width", "channel"],
                },
                # "exterior_image_2_left": {
                #     "dtype": "image",
                #     "shape": IMAGE_SHAPE,
                #     "names": ["height", "width", "channel"],
                # },
                "wrist_image_left": {
                    "dtype": "image",
                    "shape": IMAGE_SHAPE,
                    "names": ["height", "width", "channel"],
                },
                "joint_position": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["joint_position"],
                },
                "gripper_position": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["gripper_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    
    # save_dir = "wrist_images"
    # os.makedirs(save_dir, exist_ok=True)

    arg = Args()

    teleop_interface = Oculus_mobile(pos_sensitivity=0.5, rot_sensitivity=0.3,) # 1.0, 0.8
    teleop_interface.reset()
    robot = connect_to_franka()
    gripper = connect_to_gripper()
    move_to_home(robot)
    # grasp(gripper=gripper)
    release_gripper(gripper)
    last_joint_state = None
    last_gripper_state_ =None
    last_joint_state_chunk = None
    grasp_state_ = False

    TARGET_FPS = 15
    DT = 1.0 / float(TARGET_FPS)
    last_save_time = 0.0
    is_recording = False
    prev_RJ = False
    
    episode_buffer = []
    print("start teleoperation...")
    camera_reader = Camera_Read()
    chunk_number = 0

    while True:
        with torch.inference_mode():
            pose_R, RG, A, B, RJ = (teleop_interface.advance())

            pose_R = pose_R.astype("float32")
            trans = pose_R[:3]
            rot = pose_R[3:]
            quat = Rotation.from_rotvec(rot).as_quat()
            ee_pose_new = Affine(trans, quat)

            current_joint_state = robot.current_joint_state.position
            # joint_vel = robot.current_joint_state.velocity
            gripper_state = gripper.width  # [7.22333e-06, 0.08089]

            if gripper_state<0.079:
                gripper_state_ = 1
            else:
                gripper_state_ = 0
            if RG and is_recording:
                motion = CartesianMotion(ee_pose_new, ReferenceType.Relative)
                robot.move(motion)
            if A and (not grasp_state_):
                grasp(gripper)
                grasp_state_ = True
            if B and (grasp_state_):
                release_gripper(gripper)
                grasp_state_ = False
            if (not prev_RJ) and RJ:
                if not is_recording:
                    is_recording = True
                    last_save_time = 0.0
                    episode_buffer.clear()
                    # current_task = input("Enter instruction: ")
                    # current_task = "pick up the pepper"
                    current_task = "pick up the pepper"

                    print("🎬 Start recording a new episode.")
                else:
                    if input("Save? (enter y or n): ").lower() == "y":
                        for frame in episode_buffer:
                            dataset.add_frame(frame)
                        dataset.save_episode()
                        print("🎞️ Episode saved.")
                    else:
                        print("🎞️ Episode not saved.")
                        
                    episode_buffer.clear()
                    is_recording = False

                    grasp_state_ = False
                    release_gripper(gripper)
                    move_to_home(robot)
                    release_gripper(gripper)
            prev_RJ = RJ

            if is_recording and (RG or A or B) and last_joint_state is not None:
                now = time.time()
                if (now - last_save_time >= DT):
                    image = camera_reader.get_images(args=arg)
                    # frame = {
                    #     "image": image['left_image'],
                    #     "wrist_image": image['wrist_image'],
                    #     "state": np.concatenate([last_joint_state, [gripper_state_]]).astype(np.float32),
                    #     "actions": np.concatenate([current_joint_state, [gripper_state_]]).astype(np.float32),
                    #     "task": str(current_task),
                    # }
                    # print(np.asarray(last_gripper_state_, dtype=np.float32))
                    frame = {
                        "exterior_image_1_left": resize_with_pad_pil(image['left_image'], IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                        "wrist_image_left": resize_with_pad_pil(image['wrist_image'], IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                        "joint_position": np.float32(last_joint_state),
                        "gripper_position": np.asarray(last_gripper_state_, dtype=np.float32).reshape(1,),
                        "actions": np.concatenate([current_joint_state-last_joint_state_chunk, [gripper_state_]]).astype(np.float32),
                        "task": str(current_task),
                    }
                    last_joint_state = current_joint_state
                    chunk_number = chunk_number+1
                    if chunk_number==15:
                        last_joint_state_chunk = current_joint_state
                        chunk_number = 0
                    last_gripper_state_ = gripper_state_
                    episode_buffer.append(frame)
                    last_save_time = now
                    print("----------------------------")
                    # filename = os.path.join(save_dir, f"wrist_{i}.png")
                    # cv2.imwrite(filename, frame["wrist_image"])
            if is_recording and (RG or A or B) and last_joint_state is None:
                last_joint_state = current_joint_state
                last_gripper_state_ = gripper_state_
                last_joint_state_chunk = current_joint_state

if __name__ == "__main__":
    main()





# dataset = LeRobotDataset.create(
#     repo_id=REPO_NAME,
#     robot_type="panda",
#     fps=10,
#     features={
#         "image": {
#             "dtype": "image",
#             "shape": IMAGE_SHAPE, # (256, 256, 3)
#             "names": ["height", "width", "channel"],
#         },
#         "wrist_image": {
#             "dtype": "image",
#             "shape": IMAGE_SHAPE,
#             "names": ["height", "width", "channel"],
#         },
#         "state": {
#             "dtype": "float32",
#             "shape": (8,),
#             "names": ["state"],
#         },
#         "actions": {
#             "dtype": "float32",
#             "shape": (8,),
#             "names": ["actions"],
#         },
#     },
#     image_writer_threads=10,
#     image_writer_processes=5,
# )
# dataset = LeRobotDataset.create(
#     repo_id=REPO_NAME,
#     robot_type="panda",
#     fps=10,
#     features={
#         "external_image_left": {
#             "dtype": "image",
#             "shape": IMAGE_SHAPE,
#             "names": ["height", "width", "channel"],
#         },
#         "wrist_image": {
#             "dtype": "image",
#             "shape": IMAGE_SHAPE,
#             "names": ["height", "width", "channel"],
#         },
#         "joint_position": {
#             "dtype": "float64",
#             "shape": (7,),
#             "names": ["joint_position"],
#         },
#         "gripper_position": {
#             "dtype": "float64",
#             "shape": (1,),
#             "names": ["gripper_position"],
#         },
#         "actions": {
#             "dtype": "float32",
#             "shape": (8,),
#             "names": ["actions"],
#         },
#     },
#     image_writer_threads=10,
#     image_writer_processes=5,
# )