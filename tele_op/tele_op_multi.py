import argparse
import os
import time
from pathlib import Path
from typing import Optional
import multiprocessing as mp
import shutil

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from typing import Optional

import franky
from franky import *
from franky import Affine

import torch
from oculus.oculus_v0 import Oculus_mobile
from image_read import Camera_Read

# ---------------------------
# Config & Globals
# ---------------------------
HF_LEROBOT_HOME = Path("/home/yuan/VLA/tele_op/lerobot")
REPO_NAME = "lerobot_absolute_joint_center_white"
os.environ["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

IMAGE_SHAPE = (224, 224, 3)
TARGET_FPS = 15
DT = 1.0 / float(TARGET_FPS)

GRIPPER_CLOSE_THRESH = 0.079  # same logic as your original
ROBOT_IP = "9.9.9.9"          # replace with actual IP

# ---------------------------
# Utilities
# ---------------------------
def resize_with_pad_pil(image: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    image = Image.fromarray(image)
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, (height - resized_height) // 2)
    pad_width = max(0, (width - resized_width) // 2)
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.array(zero_image)

def connect_to_franka():
    robot = Robot(ROBOT_IP)
    robot.relative_dynamics_factor = 0.05
    return robot

def connect_to_gripper():
    return franky.Gripper(ROBOT_IP)

def grasp(gripper):
    speed = 0.02
    force = 20.0
    # Using move_async like your original
    gripper.move_async(0.00, speed)

def release_gripper(gripper):
    speed = 0.05
    gripper.open(speed)

def move_to_home(robot):
    robot.relative_dynamics_factor = 0.2
    q_start = np.array([-0.6, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 1.9])
    motion = JointMotion(q_start, ReferenceType.Absolute)
    robot.move(motion)
    robot.relative_dynamics_factor = 0.04

class Args:
    # Hardware parameters
    left_camera_id: str = "24285872"  # 26658469
    right_camera_id: str = None
    wrist_camera_id: str = "11022812"

    # Policy parameters
    external_camera: Optional[str] = "left"

    # Rollout parameters
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

# ---------------------------
# Inter-process protocol
# ---------------------------
# Control -> Logger commands
CMD_START_RECORDING = "START_RECORDING"   # payload: {"task": str}
CMD_STOP_RECORDING  = "STOP_RECORDING"    # payload: None
CMD_SAVE_EPISODE    = "SAVE_EPISODE"      # payload: {"save": bool}
CMD_SHUTDOWN        = "SHUTDOWN"          # payload: None

# ---------------------------
# Logger Process
# ---------------------------
def logger_process(cmd_queue: mp.Queue, recording_event: mp.Event, shutdown_event: mp.Event):
    """
    Reads camera + robot state at fixed rate and writes to LeRobotDataset.
    Responds to commands from control process for start/stop/save.
    """
    print("[Logger] Starting...")
    # Dataset setup
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        dataset = LeRobotDataset(repo_id=REPO_NAME)
        dataset.episode_buffer = dataset.create_episode_buffer()
    else:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="panda",
            fps=TARGET_FPS,
            features={
                "exterior_image_1_left": {
                    "dtype": "image",
                    "shape": IMAGE_SHAPE,
                    "names": ["height", "width", "channel"],
                },
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
                    "shape": (8,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    # Camera + robot readers (read-only)
    args = Args()
    camera_reader = Camera_Read()
    robot_reader = connect_to_franka()
    gripper_reader = connect_to_gripper()

    episode_buffer = []
    last_joint_state = None
    last_gripper_binary = None
    current_task = "pick up the pepper"  # default, will be overridden by START_RECORDING payload

    # Main loop
    next_tick = time.monotonic()
    recording = False
    frame_idx = 0

    try:
        while not shutdown_event.is_set():
            # Handle all pending commands first (non-blocking)
            try:
                while True:
                    cmd = cmd_queue.get_nowait()
                    ctype = cmd.get("type")
                    payload = cmd.get("payload", {})
                    if ctype == CMD_START_RECORDING:
                        current_task = payload.get("task", "pick up the pepper")
                        episode_buffer.clear()
                        last_joint_state = None
                        last_gripper_binary = None
                        frame_idx = 0
                        recording = True
                        recording_event.set()
                        print("[Logger] 🎬 Start recording:", current_task)
                    elif ctype == CMD_STOP_RECORDING:
                        recording = False
                        recording_event.clear()
                        print("[Logger] ⏹️ Stop recording (pending save/discard decision).")
                    elif ctype == CMD_SAVE_EPISODE:
                        save = payload.get("save", False)
                        if save and len(episode_buffer) > 0:
                            for frame in episode_buffer:
                                dataset.add_frame(frame)
                            dataset.save_episode()
                            print("[Logger] 🎞️ Episode saved. Frames:", len(episode_buffer))
                        else:
                            print("[Logger] 🎞️ Episode discarded.")
                        episode_buffer.clear()
                        last_joint_state = None
                        last_gripper_binary = None
                        frame_idx = 0
                    elif ctype == CMD_SHUTDOWN:
                        print("[Logger] Received shutdown.")
                        shutdown_event.set()
                    else:
                        print("[Logger] Unknown command:", ctype)
            except Exception:
                # queue empty or transient error; continue
                pass

            # Tick at fixed frequency
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
                continue
            next_tick += DT

            if not recording:
                continue

            # Capture
            try:
                images = camera_reader.get_images(args=args)
                left_img = resize_with_pad_pil(images["left_image"], IMAGE_SHAPE[0], IMAGE_SHAPE[1])
                wrist_img = resize_with_pad_pil(images["wrist_image"], IMAGE_SHAPE[0], IMAGE_SHAPE[1])

                current_joint_state = robot_reader.current_joint_state.position
                gripper_width = gripper_reader.width
                gripper_binary = 1 if gripper_width < GRIPPER_CLOSE_THRESH else 0

                if last_joint_state is None:
                    # Initialize previous state for actions
                    last_joint_state = current_joint_state
                    last_gripper_binary = gripper_binary
                    # Skip first frame to keep semantics same as your original
                    continue

                frame = {
                    "exterior_image_1_left": left_img,
                    "wrist_image_left": wrist_img,
                    "joint_position": np.float32(last_joint_state),
                    "gripper_position": np.asarray(last_gripper_binary, dtype=np.float32).reshape(1,),
                    "actions": np.concatenate([current_joint_state, [gripper_binary]]).astype(np.float32),
                    "task": str(current_task),
                }

                episode_buffer.append(frame)
                last_joint_state = current_joint_state
                last_gripper_binary = gripper_binary
                frame_idx += 1
                print(f"[Logger] Frame {frame_idx}")

            except Exception as e:
                print("[Logger] Capture error:", e)

    except KeyboardInterrupt:
        print("[Logger] KeyboardInterrupt.")
    finally:
        print("[Logger] Exiting...")
        # No dataset.save here; control decides via SAVE_EPISODE

# ---------------------------
# Control Process (Main)
# ---------------------------
def control_process(cmd_queue: mp.Queue, recording_event: mp.Event, shutdown_event: mp.Event):
    """
    Handles teleoperation and robot commands. Tells logger when to start/stop and whether to save.
    """
    print("[Control] Starting teleoperation...")
    teleop_interface = Oculus_mobile(pos_sensitivity=0.8, rot_sensitivity=0.5)
    teleop_interface.reset()

    robot = connect_to_franka()
    gripper = connect_to_gripper()

    move_to_home(robot)
    release_gripper(gripper)

    prev_RJ = False
    grasp_state = False
    is_recording = False
    current_task = "pick up the pepper"  # your default

    try:
        with torch.inference_mode():
            while not shutdown_event.is_set():
                pose_R, RG, A, B, RJ = teleop_interface.advance()

                pose_R = pose_R.astype("float32")
                trans = pose_R[:3]
                rot = pose_R[3:]
                quat = Rotation.from_rotvec(rot).as_quat()
                ee_pose_new = Affine(trans, quat)

                # Robot state (for control only; logger reads its own robot)
                current_joint_state = robot.current_joint_state.position
                gripper_width = gripper.width
                gripper_binary = 1 if gripper_width < GRIPPER_CLOSE_THRESH else 0

                # Motion gating by RG (like your original)
                if RG and is_recording:
                    motion = CartesianMotion(ee_pose_new, ReferenceType.Relative)
                    robot.move(motion, asynchronous=True)

                # Grasp/open on A/B press (as in original)
                if A and (not grasp_state):
                    grasp(gripper)
                    grasp_state = True
                if B and grasp_state:
                    release_gripper(gripper)
                    grasp_state = False

                # RJ rising edge: toggle recording start/stop
                if (not prev_RJ) and RJ:
                    if not is_recording:
                        # Start a new episode
                        is_recording = True
                        # You can change this to get task from user if needed
                        current_task = "pick up the pepper"
                        cmd_queue.put({"type": CMD_START_RECORDING, "payload": {"task": current_task}})
                        print("[Control] 🎬 Start recording a new episode.")
                    else:
                        # Stop and ask to save/discard
                        cmd_queue.put({"type": CMD_STOP_RECORDING})
                        print("[Control] Save? (press A to save, B to discard)")
                        # Wait for A/B decision
                        while True:
                            if shutdown_event.is_set():
                                break
                            _, _, A2, B2, _ = teleop_interface.advance()
                            if A2:
                                cmd_queue.put({"type": CMD_SAVE_EPISODE, "payload": {"save": True}})
                                print("[Control] 🎞️ Episode saved.")
                                break
                            elif B2:
                                cmd_queue.put({"type": CMD_SAVE_EPISODE, "payload": {"save": False}})
                                print("[Control] 🎞️ Episode discarded.")
                                break

                        # Reset states
                        is_recording = False
                        grasp_state = False
                        release_gripper(gripper)
                        move_to_home(robot)
                        release_gripper(gripper)

                prev_RJ = RJ

    except KeyboardInterrupt:
        print("[Control] KeyboardInterrupt.")
    finally:
        print("[Control] Exiting...")
        # Tell logger to shutdown
        cmd_queue.put({"type": CMD_SHUTDOWN})

# ---------------------------
# Entry point
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Teleoperation with multiprocessing logger.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor (unused, kept for compat).")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    # IPC primitives
    cmd_queue: mp.Queue = mp.Queue()
    recording_event = mp.Event()
    shutdown_event = mp.Event()

    # Launch logger first so it's ready
    lp = mp.Process(target=logger_process, args=(cmd_queue, recording_event, shutdown_event), daemon=True)
    lp.start()

    try:
        # Run control in the main process (so Ctrl+C works nicely)
        control_process(cmd_queue, recording_event, shutdown_event)
    finally:
        shutdown_event.set()
        cmd_queue.put({"type": CMD_SHUTDOWN})
        lp.join(timeout=5.0)
        if lp.is_alive():
            lp.terminate()

if __name__ == "__main__":
    main()
