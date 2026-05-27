"""
run_vla_inference.py

Lerobot-side client that drives an SO-ARM 101 with an OpenVLA-OFT
inference server (see vla-scripts/deploy.py).

Loop per instruction:
    capture top + wrist images + joint positions
        -> POST /act -> receive a (10, 6) action chunk
        -> execute the first --chunk-steps steps on the arm
        -> re-query
Press Enter mid-episode to abort and prompt for a new instruction.
Submit an empty instruction to quit.

Run on the lerobot conda env after opening an SSH tunnel:

    # terminal 1 (tunnel)
    ssh -L 8777:localhost:8777 <user>@<server>

    # terminal 2 (this script)
    conda activate lerobot
    python run_vla_inference.py \
        --server-url http://localhost:8777 \
        --robot-port /dev/ttyACM0 \
        --top-camera 0 --wrist-camera 1
"""

import argparse
import json
import select
import sys
import time

import json_numpy
import numpy as np
import requests

# Patch the global json module so json.dumps / json.loads handle ndarrays.
# This matches what deploy.py does on the server side.
json_numpy.patch()


from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
# SO-101 6-DOF joint order — must match the order in the training dataset.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def build_payload(robot_obs: dict, top_key: str, wrist_key: str, instruction: str) -> dict:
    state = np.asarray([robot_obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
    full = robot_obs[top_key]

    # Single-camera fallback: reuse the top image as wrist_image.
    # Server was launched with --num_images_in_input 2, so it still expects both keys.
    wrist = robot_obs[wrist_key] if wrist_key in robot_obs else full

    return {
        "full_image": full,
        "wrist_image": wrist,
        "state": state,
        "instruction": instruction,
    }


def action_dict(action_vec: np.ndarray) -> dict:
    return {f"{j}.pos": float(action_vec[i]) for i, j in enumerate(JOINTS)}


def stdin_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0])


def call_server(endpoint: str, payload: dict, timeout: float) -> np.ndarray:
    body = json.dumps(payload)  # patched json handles ndarrays
    resp = requests.post(
        endpoint, data=body, headers={"Content-Type": "application/json"}, timeout=timeout
    )
    resp.raise_for_status()
    decoded = json.loads(resp.text)  # patched -> ndarray or list of ndarrays
    arr = np.asarray(decoded, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://localhost:8777")
    p.add_argument("--robot-port", default="/dev/ttyACM1")
    p.add_argument("--top-camera", type=int, default=0)
    p.add_argument("--wrist-camera", type=int, default=1)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument(
        "--chunk-steps",
        type=int,
        default=10,
        help="Steps of each returned chunk to execute before re-querying (<=10 for SO-101).",
    )
    p.add_argument("--control-hz", type=float, default=15.0)
    p.add_argument("--request-timeout", type=float, default=15.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Query the server and print actions but do not move the robot.",
    )
    args = p.parse_args()

    endpoint = args.server_url.rstrip("/") + "/act"
    period = 1.0 / args.control_hz
    chunk_steps = max(1, args.chunk_steps)

    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=args.top_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
    }
    robot_cfg = SO101FollowerConfig(port=args.robot_port, cameras=cameras, id="my_awesome_follower_arm")
    robot = SO101Follower(robot_cfg)
    robot.connect()
    print(f"[client] robot connected on {args.robot_port}")
    print(f"[client] server endpoint: {endpoint}")
    if args.dry_run:
        print("[client] DRY RUN — actions will be printed, not sent to the arm")

    try:
        while True:
            try:
                instruction = input("\nInstruction (empty + Enter to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not instruction:
                break
            print(
                f"[episode] '{instruction}' — up to {args.max_steps} steps, "
                "press Enter to abort"
            )

            step = 0
            aborted = False
            t_episode = time.time()
            while step < args.max_steps and not aborted:
                obs = robot.get_observation()
                payload = build_payload(obs, "top", "wrist", instruction)

                t_query = time.time()
                try:
                    chunk = call_server(endpoint, payload, args.request_timeout)
                except (requests.RequestException, ValueError) as e:
                    print(f"[error] server call failed: {e}")
                    break
                query_ms = (time.time() - t_query) * 1000

                n = min(chunk_steps, chunk.shape[0])
                if step == 0:
                    print(
                        f"[debug] chunk shape={chunk.shape}, "
                        f"first action={np.round(chunk[0], 3).tolist()}, "
                        f"server latency={query_ms:.0f} ms"
                    )

                for k in range(n):
                    t0 = time.time()
                    if not args.dry_run:
                        robot.send_action(action_dict(chunk[k]))
                    step += 1
                    if stdin_pressed():
                        sys.stdin.readline()
                        aborted = True
                        print(f"[episode] aborted by user at step {step}")
                        break
                    sleep_for = period - (time.time() - t0)
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            if not aborted:
                print(f"[episode] finished at step {step} ({time.time() - t_episode:.1f}s)")
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[warn] disconnect failed: {e}")
        print("[client] done")


if __name__ == "__main__":
    main()
