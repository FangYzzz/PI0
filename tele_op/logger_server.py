#!/usr/bin/env python3
"""
ZeroRPC Logger Server (push-state, async save, optional images)
- Only opens cameras if SAVE_IMAGES is True.
- Receives robot state via RPC `push_state(joint_position, gripper_width, ts)` from control client.
- Captures frames at TARGET_FPS using the latest pushed state and writes to LeRobotDataset.
- `save_episode(True)` returns immediately; the heavy disk I/O runs in a background greenlet so heartbeats don't time out.

Run:
    python logger_server_push_v2.py --bind tcp://0.0.0.0:4242

IMPORTANT: We monkey-patch gevent BEFORE other imports in the server process.
"""

from gevent import monkey  # must be first
monkey.patch_socket()
monkey.patch_select()
import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

import gevent
import zerorpc

# Import IO after patch
try:
    from image_read import Camera_Read
except Exception:
    Camera_Read = None  # allow running without camera if SAVE_IMAGES is False

from typing import Optional
from openpi_client import image_tools

# ---------------------------
# Config
# ---------------------------
HF_LEROBOT_HOME = Path("/home/yuan/VLA/tele_op/lerobot")
REPO_NAME = "lerobot_absolute_joint_center_white222"
os.environ["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

IMAGE_SHAPE = (224, 224, 3)
TARGET_FPS = 15
GRIPPER_CLOSE_THRESH = 0.079
SAVE_IMAGES = True  # << set False to avoid saving images to disk

# ---------------------------
# Utils
# ---------------------------
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

arg = Args()
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
    return np.array(zero_image)

# ---------------------------
# RPC Server Impl
# ---------------------------
class LoggerRPC(object):
    def __init__(self, fps: int = TARGET_FPS):
        output_path = HF_LEROBOT_HOME / REPO_NAME
        # Dataset features (conditionally include images)
        if output_path.exists():
            
            self.dataset = LeRobotDataset(repo_id=REPO_NAME)
            # Start a fresh buffer for the next episode index (meta.total_episodes)
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
            # shutil.rmtree(output_path)
        else:    
            print("create new dataset !!!!!!!!!!!!!!!!",output_path)
            self.dataset = LeRobotDataset.create(
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

        # Cameras only if saving images
        self.camera_reader = Camera_Read() 

        # State
        self.recording = False
        self.current_task = "pick up the pepper"
        self.episode_buffer = []
        self.last_joint_state = None
        self.last_gripper_binary = None
        self._latest_state = None  # dict: {"q": np.float32[7], "gripper_width": float, "ts": float}
        self.frame_idx = 0

        # Loop
        self.dt = 1.0 / float(fps)
        self._stop = gevent.event.Event()
        self._loop_greenlet = None
        self._save_greenlet = None
        self.saving = False
        self._last_used_ts = None
        self.last_time = None
    # ---- RPC methods ----
    def ping(self):
        return {"status": "ok", "recording": self.recording, "frames": self.frame_idx}

    def push_state(self, joint_position, gripper_width, ts):
        if self.saving:
            print("saving")
            return {"ok": True}
        try:

            q = np.asarray(joint_position, dtype=np.float32)
            if q.shape != (7,):
                return {"ok": False, "error": f"bad q shape {q.shape}"}
            self._latest_state = {"q": q, "gripper_width": float(gripper_width), "ts": float(ts)}
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def start_recording(self, task: str = "pick up the pepper"):
        self.current_task = str(task)
        self.episode_buffer.clear()
        self.last_joint_state = None
        self.last_gripper_binary = None
        self.frame_idx = 0
        self.recording = True
        self._spawn_capture()
        print("[Logger] 🎬 Start recording:", self.current_task)
        if self.saving:
            return {"ok": False, "error": "busy_saving"}
        return {"ok": True}

    def stop_recording(self):
        self.recording = False
        print("[Logger] ⏹️ Stop recording (awaiting save/discard)")
        return {"ok": True}

    def _spawn_capture(self):
        if self._loop_greenlet is None or self._loop_greenlet.dead:
            self._stop.clear()
            self._loop_greenlet = gevent.spawn(self._capture_loop)
            print("[Logger] capture loop SPAWNED", flush=True)

    def _kill_capture(self, timeout=2.0):
        if self._loop_greenlet is not None and not self._loop_greenlet.dead:
            self._stop.set()
            try:
                self._loop_greenlet.join(timeout=timeout)
            except Exception:
                pass
            if not self._loop_greenlet.dead:
                self._loop_greenlet.kill()
            print("[Logger] capture loop KILLED", flush=True)
        self._loop_greenlet = None

    def _save_worker(self, frames):
        try:
            for i, frame in enumerate(frames):
                self.dataset.add_frame(frame)
                print('i')
                # if (i % 50) == 0:
                #     gevent.sleep(0)  # 让出循环，避免心跳丢失
            self.dataset.save_episode()
            print(f"[Logger] 🎞️ Background save finished. Frames: {len(frames)}")
        except Exception as e:
            print("[Logger] Save worker error:", repr(e))
        finally:
            self.saving = False  # ← 保存结束，解除暂停

    def save_episode(self, save: bool):
        save = bool(save)
        if save and len(self.episode_buffer) > 0:

            # frames_to_save = self.episode_buffer
            # # self.episode_buffer = []
            # # kick off background save so RPC returns immediately
            # self.saving = True
            # self.recording = False
            # self._save_greenlet = gevent.spawn(self._save_worker, frames_to_save)
            # self._kill_capture()
            # # for i, frame in enumerate(frames_to_save):
            # #     self.dataset.add_frame(frame)
            # #     # if (i % 50) == 0:
            # #         # gevent.sleep(0)  # 让出循环，避免心跳丢失
            # # self.dataset.save_episode()
            # print("[Logger] ⏳ Background saving started...", flush=True)

            # self.episode_buffer.clear()
            self.saving=True
        else:
            print("[Logger] 🎞️ Episode discarded.")
            self.episode_buffer.clear()
        # reset counters regardless
        self.last_joint_state = None
        self.last_gripper_binary = None
        self.frame_idx = 0
        return {"ok": True}

    def shutdown(self):
        print("[Logger] Shutdown requested.")
        self._stop.set()
        try:
            if self._save_greenlet is not None:
                self._save_greenlet.join(timeout=2.0)
            self._loop_greenlet.join(timeout=2.0)
        except Exception:
            pass
        return {"ok": True}

    # ---- Capture loop ----
    def _capture_loop(self):
        
        next_tick = time.monotonic()
        print("[Logger] capture loop START", flush=True)
        while not self._stop.is_set():
            if self.saving:
                for i, frame in enumerate(self.episode_buffer):
                    self.dataset.add_frame(frame)
                    print('i')
                    # if (i % 50) == 0:
                    #     gevent.sleep(0)  # 让出循环，避免心跳丢失
                self.dataset.save_episode()
                self.episode_buffer.clear()
                self.saving=False
            now = time.monotonic()
            if now < next_tick:
                gevent.sleep(max(0.0, next_tick - now))
                continue
            next_tick += self.dt

            if not self.recording:
                gevent.sleep(0)
                continue
            
            try:
                time_now = time.time()
                # Need state to proceed
                state = self._latest_state
                if state is None:
                    continue
                
                ts = state["ts"]
                if self._last_used_ts is not None and ts is not None and ts <= self._last_used_ts:
                # 不是新状态，跳过
                    continue
                images = self.camera_reader.get_images(args=arg)
                current_joint_state = state["q"]
                gripper_width = state["gripper_width"]
                gripper_binary = 1 if gripper_width < GRIPPER_CLOSE_THRESH else 0

                if self.last_joint_state is None:
                    self.last_joint_state = current_joint_state
                    self.last_gripper_binary = gripper_binary
                    continue

                frame = {
                    "exterior_image_1_left": image_tools.resize_with_pad(images["left_image"], IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                    "wrist_image_left": image_tools.resize_with_pad(images["wrist_image"], IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
                    "joint_position": np.float32(self.last_joint_state),
                    "gripper_position": np.asarray(self.last_gripper_binary, dtype=np.float32).reshape(1,),
                    "actions": np.concatenate([current_joint_state, [gripper_binary]]).astype(np.float32),
                    "task": str(self.current_task),
                }
               

                self.episode_buffer.append(frame)
                self.last_joint_state = current_joint_state
                self.last_gripper_binary = gripper_binary
                self.frame_idx += 1
                self._last_used_ts = ts
                if (self.frame_idx % 1) == 0:
                    print(f"[Logger] Frame {self.frame_idx}")
                if self.last_time is not None:
                    print(time_now-self.last_time)
                self.last_time = time_now
            except Exception as e:
                print("[Logger] Capture error:", repr(e), flush=True)
                gevent.sleep(self.dt)

# ---------------------------
# Runner
# ---------------------------

def run_server(bind_url: str):
    # Increase heartbeat to tolerate brief pauses
    srv = zerorpc.Server(LoggerRPC(), heartbeat=10)
    srv.bind(bind_url)
    print(f"[Logger] ZeroRPC server listening on {bind_url}")
    try:
        srv.run()
    except KeyboardInterrupt:
        print("[Logger] Server interrupted.")


def main():
    parser = argparse.ArgumentParser(description="ZeroRPC Logger Server (push-state, async save)")
    parser.add_argument("--bind", default="tcp://0.0.0.0:4242", help="Bind URL")
    args = parser.parse_args()
    run_server(args.bind)


if __name__ == "__main__":
    main()
