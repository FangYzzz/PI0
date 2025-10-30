#!/usr/bin/env python3
"""
ZeroRPC Control Client (decoupled state publisher)
- Owns the ONLY Franka connection (robot + gripper).
- Samples robot state in a FAST background thread (e.g., 200 Hz).
- Pushes the latest state to the logger in a SEPARATE thread (e.g., 80–100 Hz),
  with its OWN ZeroRPC client created inside that thread (gevent hub safe).
- Main loop only handles Oculus + robot motion + start/stop/save RPCs.

Run:
    python control_client.py --connect tcp://127.0.0.1:4242
"""

import argparse
import time
import threading
from typing import Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Robot & IO
import franky
from franky import *
from franky import Affine
from oculus.oculus_v0 import Oculus_mobile

GRIPPER_CLOSE_THRESH = 0.079
ROBOT_IP = "9.9.9.9"

def connect_to_franka():
    robot = Robot(ROBOT_IP)
    robot.relative_dynamics_factor = 0.05
    return robot

def connect_to_gripper():
    return franky.Gripper(ROBOT_IP)

def grasp(gripper):
    speed = 0.02
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

class LatestState:
    def __init__(self):
        self._lock = threading.Lock()
        self.q = None
        self.w = None
        self.ts = 0.0

    def update(self, q: np.ndarray, width: float, ts: float):
        with self._lock:
            self.q = q
            self.w = width
            self.ts = ts

    def snapshot(self):
        with self._lock:
            q = None if self.q is None else self.q.copy()
            return q, self.w, self.ts

def start_state_threads(connect_url: str, robot, gripper, sample_hz=200.0, push_hz=200.0) -> Tuple[threading.Event, threading.Thread, threading.Thread]:
    """
    Start two background threads:
      - sampler: poll robot at high rate and store latest state
      - pusher: push latest state to logger at push_hz, using its OWN ZeroRPC client created in-thread
    """
    latest = LatestState()
    stop = threading.Event()

    def sampler():
        period = 1.0 / float(sample_hz)
        next_t = time.perf_counter()
        while not stop.is_set():
            try:
                q = robot.current_joint_state.position
                w = float(gripper.width)
                latest.update(q, w, time.time())
            except Exception:
                pass
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    def pusher():
        # create RPC client IN THIS THREAD (gevent hub safety)
        import zerorpc
        cli = zerorpc.Client(timeout=10, heartbeat=10)
        cli.connect(connect_url)
        try:
            period = 1.0 / float(push_hz)
            next_t = time.perf_counter()
            last_ts = 0.0
            while not stop.is_set():
                q, w, ts = latest.snapshot()
                if q is not None and w is not None and ts > last_ts:
                    try:
                        cli.push_state(q.tolist(), float(w), ts)
                        last_ts = ts
                    except Exception:
                        pass
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
        finally:
            try:
                cli.close()
            except Exception:
                pass

    th_sampler = threading.Thread(target=sampler, name="robot_sampler", daemon=True)
    th_pusher  = threading.Thread(target=pusher,  name="state_pusher", daemon=True)
    th_sampler.start()
    th_pusher.start()
    return stop, th_sampler, th_pusher

def run_control(connect_url: str):
    # Oculus first
    teleop = Oculus_mobile(pos_sensitivity=0.8, rot_sensitivity=0.5)
    teleop.reset()

    print("[Control] Checking Oculus tracking...")
    _ok = False
    for _ in range(300):
        pose_R, RG, A, B, RJ = teleop.advance()
        if pose_R is not None and pose_R.any():
            _ok = True
            break
        time.sleep(0.03)
    if not _ok:
        print("[Control] ⚠️ No valid Oculus transforms after warmup; continuing anyway.")

    # Control RPC client (main thread only)
    import zerorpc
    rpc = zerorpc.Client(timeout=10, heartbeat=10)
    rpc.connect(connect_url)

    robot = connect_to_franka()
    gripper = connect_to_gripper()
    move_to_home(robot)
    release_gripper(gripper)

    # Start decoupled state sampler + pusher (pusher creates its own RPC client)
    stop_event, th_sampler, th_pusher = start_state_threads(connect_url, robot, gripper, sample_hz=200.0, push_hz=80.0)

    prev_RJ = False
    grasp_state = False
    is_recording = False
    current_task = "pick up the pepper"

    try:
        with torch.inference_mode():
            while True:
                pose_R, RG, A, B, RJ = teleop.advance()

                pose_R = pose_R.astype("float32")
                trans = pose_R[:3]
                rot = pose_R[3:]
                quat = Rotation.from_rotvec(rot).as_quat()
                ee_pose_new = Affine(trans, quat)

                if RG and is_recording:
                    motion = CartesianMotion(ee_pose_new, ReferenceType.Relative)
                    robot.move(motion, asynchronous=True)

                if A and (not grasp_state):
                    grasp(gripper)
                    grasp_state = True
                if B and grasp_state:
                    release_gripper(gripper)
                    grasp_state = False

                if (not prev_RJ) and RJ:
                    # ensure server reachable
                    try:
                        _ = rpc.ping()
                    except Exception as e:
                        try:
                            rpc = __import__("zerorpc").Client(timeout=10, heartbeat=10)
                            rpc.connect(connect_url)
                        except Exception:
                            print("[Control] RPC reconnect failed:", e)

                    if not is_recording:
                        is_recording = True
                        current_task = "pick up the pepper"
                        rpc.start_recording(current_task)
                        print("[Control] 🎬 Start recording a new episode.")
                    else:
                        rpc.stop_recording()
                        print("[Control] Save? (press A to save, B to discard)")
                        while True:
                            _, _, A2, B2, _ = teleop.advance()
                            if A2:
                                try:
                                    rpc.save_episode(True)  # background save on server
                                except Exception:
                                    pass
                                print("[Control] 🎞️ Episode save requested.")
                                break
                            elif B2:
                                try:
                                    rpc.save_episode(False)
                                except Exception:
                                    pass
                                print("[Control] 🎞️ Episode discarded.")
                                break
                        is_recording = False
                        grasp_state = False
                        release_gripper(gripper)
                        move_to_home(robot)
                        release_gripper(gripper)

                prev_RJ = RJ

    except KeyboardInterrupt:
        print("[Control] KeyboardInterrupt.")
    finally:
        # stop background threads
        try:
            stop_event.set()
            th_sampler.join(timeout=1.0)
            th_pusher.join(timeout=1.0)
        except Exception:
            pass
        # close main-thread RPC client (don't call remote shutdown)
        try:
            rpc.close()
        except Exception:
            pass
        print("[Control] Exiting.")

def main():
    parser = argparse.ArgumentParser(description="ZeroRPC Control Client (decoupled state publisher)")
    parser.add_argument("--connect", default="tcp://127.0.0.1:4242", help="Server connect URL")
    args = parser.parse_args()
    run_control(args.connect)

if __name__ == "__main__":
    main()
