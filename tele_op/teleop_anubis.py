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
from openpi_client import image_tools

HF_LEROBOT_HOME = Path("/home/yuan/VLA/tele_op/lerobot")
REPO_NAME = "lerobot_absolute_quat_uniform_white_flange_pnp" # (with pause)  # lerobot_pickup_pepper_absolute_joint(without pause)
os.environ["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


IMAGE_SHAPE = (224, 224, 3)   
DYNAMIC_FACTOR = 0.06

# def resize_with_pad_pil(image: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
#     """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
#     width without distortion by padding with zeros.

#     Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
#     """
#     image = Image.fromarray(image)
    
#     cur_width, cur_height = image.size
#     if cur_width == width and cur_height == height:
#         return image  # No need to resize if the image is already the correct size.

#     ratio = max(cur_width / width, cur_height / height)
#     resized_height = int(cur_height / ratio)
#     resized_width = int(cur_width / ratio)
#     resized_image = image.resize((resized_width, resized_height), resample=method)

#     zero_image = Image.new(resized_image.mode, (width, height), 0)
#     pad_height = max(0, int((height - resized_height) / 2))
#     pad_width = max(0, int((width - resized_width) / 2))
#     zero_image.paste(resized_image, (pad_width, pad_height))
#     assert zero_image.size == (width, height)
#     zero_image = np.array(zero_image)
#     return zero_image

def connect_to_franka():
    robot = Robot("9.9.9.9") 
    robot.relative_dynamics_factor = DYNAMIC_FACTOR

    return robot

def connect_to_gripper():
    gripper = franky.Gripper("9.9.9.9")
    
    return gripper

def grasp(gripper):
    speed = 0.02  # 0.02[m/s]
    force = 20.0  # 20.0[N]
    # success = gripper.grasp(0.0, speed, force, epsilon_outer=1.0,asynchronous=True)
    gripper.move_async(0.00, speed)
    return 1
    # Move the fingers to a specific width (5cm)
    # success = gripper.move(0.05, speed)

    # Wait for 5s
    # success_future = gripper.move_async(0.05, speed)
    # success_future.wait(5)

def release_gripper(gripper):
    speed = 0.05  # 0.02[m/s]
    gripper.open(speed)
    return 0

def move_to_home(robot):
    robot.relative_dynamics_factor = 0.2
    # q_start = np.array([0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398])
    q_start = np.array([-0.6, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
    motion = JointMotion(q_start, ReferenceType.Absolute)
    robot.move(motion, asynchronous=True)
    # robot.relative_dynamics_factor = DYNAMIC_FACTOR  # 0.05

def move_to_home_syn(robot):
    robot.relative_dynamics_factor = 0.2
    # q_start = np.array([0.0, -0.785398, 0.0, -2.35619, 0.0, 1.5708, 0.785398])
    q_start = np.array([-0.6, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
    motion = JointMotion(q_start, ReferenceType.Absolute)
    robot.move(motion, asynchronous=False)
    # robot.relative_dynamics_factor = DYNAMIC_FACTOR # 0.05

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
                "eef_position": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["eef_position"],
                },
                "gripper_position": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["gripper_position"],
                },
                "joint_actions": {
                    "dtype": "float32",
                    "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                    "names": ["joint_actions"],
                },
                "eef_actions": {
                    "dtype": "float32",
                    "shape": (7,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                    "names": ["eef_actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    
    # save_dir = "wrist_images"
    # os.makedirs(save_dir, exist_ok=True)

    arg = Args()

    teleop_interface = Oculus_mobile(pos_sensitivity=4, rot_sensitivity=1,) # 1.0, 0.8
    teleop_interface.reset()
    robot = connect_to_franka()
    gripper = connect_to_gripper()
    move_to_home(robot)
    # grasp(gripper=gripper)
    release_gripper(gripper)
    last_joint_state = None
    last_gripper_state_ =None
    grasp_state_ = False
    last_eef_state =None
    gripper_state_ = 0
    TARGET_FPS = 15
    DT = 1.0 / float(TARGET_FPS)
    last_save_time = 0.0
    is_recording = False
    prev_RJ = False
    
    episode_buffer = []
    print("start teleoperation...")
    camera_reader = Camera_Read()
    i = 0
    image = camera_reader.get_images(args=arg)
    robot.relative_dynamics_factor = DYNAMIC_FACTOR
    while True:
        
        with torch.inference_mode():
            pose_R, RG, A, B, RJ, RTr = (teleop_interface.advance())
            
            pose_R = pose_R.astype("float32")
            trans = pose_R[:3]
            rot = pose_R[3:]
            quat = Rotation.from_rotvec(rot).as_quat()
            ee_pose_new = Affine(trans, quat)
            # time_start = time.time()
            # print(gripper.width)
            # current_joint_state = robot.current_joint_state.position
            # time_start = time.time()
            # joint_vel = robot.current_joint_state.velocity
            # gripper_state = gripper  # [  7.22333e-06, 0.08089]
            # time_end = time.time()
            # print("time use readr:", time_end-time_start)

            # if gripper_state<0.079:
            #     gripper_state_ = 1
            # else:
            #     gripper_state_ = 0
        
            
            if (not prev_RJ) and RJ:
                if not is_recording:
                    is_recording = True
                    last_save_time = 0.0
                    episode_buffer.clear()
                    # current_task = input("Enter instruction: ")
                    # current_task = "pick up the pepper"
                    current_task = "pick up the pepper and place it in the bowl"

                    print("🎬 Start recording a new episode.")
                else:
                    # if input("Save? (enter y or n): ").lower() == "y":
                    #     for frame in episode_buffer:
                    #         dataset.add_frame(frame)
                    #     dataset.save_episode()
                    #     print("🎞️ Episode saved.")
                    # else:
                    #     print("🎞️ Episode not saved.")
                    print("Save? (enter A or B): ")
                    i=0
                    while True:
                        pose_R, RG, A, B, RJ, RTr = (teleop_interface.advance())
                        if A:
                            print("🎞️ Episode saving---.")
                            gripper_state_ = release_gripper(gripper)
                            move_to_home(robot)
                            for frame in episode_buffer:
                                dataset.add_frame(frame)
                            dataset.save_episode()
                            print("🎞️ Episode saved.")
                            # robot.relative_dynamics_factor = DYNAMIC_FACTOR
                            break
                        elif B:
                            gripper_state_ = release_gripper(gripper)
                            move_to_home_syn(robot)
                            print("🎞️ Episode not saved.")
                            break
                        
                    episode_buffer.clear()
                    is_recording = False
                    grasp_state_ = False
                   
                    robot.relative_dynamics_factor = DYNAMIC_FACTOR
                    # release_gripper(gripper)
            prev_RJ = RJ
        if not is_recording:
            continue

        if RG and is_recording:
            motion = CartesianMotion(ee_pose_new, ReferenceType.Relative)
            robot.move(motion, asynchronous=True)
        if RTr and (not grasp_state_):
            gripper_state_ = grasp(gripper)
            grasp_state_ = True
        if B and (grasp_state_):
            gripper_state_ = release_gripper(gripper)
            grasp_state_ = False

        if is_recording and (RG or RTr or B) and last_joint_state is not None:
            now = time.time()
            # print((now - last_save_time)>DT)
            if (now - last_save_time<DT):
                # print("sleep")
                time.sleep(DT - (now - last_save_time))
            # else:
            #     print("low freq!!!!!!",now - last_save_time)

            # if (now - last_save_time >= DT):
            current_joint_state = robot.current_joint_state.position
            current_eef_state_ = robot.current_cartesian_state.pose.end_effector_pose
            # print(current_eef_state_)
            eef_t = current_eef_state_.translation
            eef_q = current_eef_state_.quaternion
            r = Rotation.from_quat(eef_q)
            rpy = r.as_euler('xyz', degrees=False)
            current_eef_state = np.concatenate([eef_t, rpy])
            # print(current_eef_state)
            # print(now)
            # print(now - last_save_time)
            i += 1 
            # start = time.time()
            image = camera_reader.get_images(args=arg)
            # end = time.time()
            # print(end - start)
            # frame = {
            #     "image": image['left_image'],
            #     "wrist_image": image['wrist_image'],
            #     "state": np.concatenate([last_joint_state, [gripper_state_]]).astype(np.float32),
            #     "actions": np.concatenate([current_joint_state, [gripper_state_]]).astype(np.float32),
            #     "task": str(current_task),
            # }
            # print(np.asarray(last_gripper_state_, dtype=np.float32))
            # print("gripper: ",grasp_state_)
            frame = {
                # image
                "exterior_image_1_left": image_tools.resize_with_pad(image['left_image'], IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                "wrist_image_left": image_tools.resize_with_pad(image['wrist_image'], IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                # state
                "joint_position": np.float32(last_joint_state),
                "eef_position": np.float32(last_eef_state),
                "gripper_position": np.asarray(last_gripper_state_, dtype=np.float32).reshape(1,),
                # actions
                "joint_actions": np.concatenate([current_joint_state, [gripper_state_]]).astype(np.float32),
                "eef_actions": np.concatenate([current_eef_state, [gripper_state_]]).astype(np.float32),
                # task
                "task": str(current_task),
            }
            # print(frame["eef_position"])
            # gripper_action = frame["eef_actions"][-1]
            last_joint_state = current_joint_state
            last_eef_state = current_eef_state
            last_gripper_state_ = gripper_state_
            episode_buffer.append(frame)
            last_save_time = now
            print(f"--------------{i}")
            # print(gripper_state_)
            # filename = os.path.join(save_dir, f"wrist_{i}.png")
            # cv2.imwrite(filename, frame["wrist_image"])
    
        if is_recording and (RG or RTr or B) and last_joint_state is None:
            current_joint_state = robot.current_joint_state.position
            current_eef_state_ = robot.current_cartesian_state.pose.end_effector_pose
            # print(current_eef_state_)
            eef_t = current_eef_state_.translation
            eef_q = current_eef_state_.quaternion
            r = Rotation.from_quat(eef_q)
            rpy = r.as_euler('xyz', degrees=False)
            current_eef_state = np.concatenate([eef_t, rpy])
            last_eef_state = current_eef_state
            last_joint_state = robot.current_joint_state.position
            last_gripper_state_ = gripper_state_

if __name__ == "__main__":
    main()




