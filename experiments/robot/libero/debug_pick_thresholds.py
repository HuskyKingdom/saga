"""
debug_pick_thresholds.py

运行一个 episode，每步打印 pick 检测相关的原始传感器值：
  - gripper_qpos[0]       夹爪开合度（闭合时接近 0）
  - EEF 到各物体的距离    用于确定 prox_thresh
  - 各物体 z 相对初始值的偏移  用于确定 lift_thresh

输出会在终端实时显示，同时保存到 debug_pick_thresholds.log。
看到 gripper 开始闭合、某个物体 z 开始升高的那几步，就是 pick 发生的时刻。

用法：
    python experiments/robot/libero/debug_pick_thresholds.py \
        --pretrained_checkpoint ckpt/ckpoints/<name> \
        --task_suite_name libero_spatial \
        --perturbation_mode none \
        --use_l1_regression False \
        --use_proprio False \
        --num_images_in_input 1 \
        --unnorm_key libero_spatial \
        --task_id 0          # 查看第几个 task（默认 0）
        --use_bddl_language True
"""

import os
import sys
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import draccus
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DebugConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = False
    use_diffusion: bool = False
    use_film: bool = False
    auto_regression: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # Required by get_vla_action internals
    remove_wrap: bool = False
    h_decoding: bool = False
    e_decoding: bool = False
    energy_alpha: float = 0.5
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50

    task_suite_name: str = "libero_spatial"
    perturbation_mode: str = "none"
    task_id: int = 0          # 只跑这一个 task
    episode_idx: int = 0      # 用第几个 initial state
    num_steps_wait: int = 10
    max_steps: int = 120      # 跑多少步就停（不需要跑完整任务）
    env_img_res: int = 256

    use_bddl_language: bool = False
    seed: int = 7


def _build_moveable_body_map(sim):
    mapping = {}
    for jnt_id, jnt_type in enumerate(sim.model.jnt_type):
        if jnt_type != 0:
            continue
        body_id = sim.model.jnt_bodyid[jnt_id]
        body_name = sim.model.body_id2name(body_id)
        if "robot" in body_name.lower():
            continue
        mapping[body_name] = body_id
    return mapping


def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_r = resize_image_for_policy(img, resize_size)
    wrist_r = resize_image_for_policy(wrist_img, resize_size)
    return {
        "full_image": img_r,
        "wrist_image": wrist_r,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


@draccus.wrap()
def debug_thresholds(cfg: DebugConfig) -> None:
    assert cfg.pretrained_checkpoint, "--pretrained_checkpoint is required"
    set_seed_everywhere(cfg.seed)

    # Check unnorm key
    model_obj = get_model(cfg)
    key = cfg.unnorm_key or cfg.task_suite_name
    if key not in model_obj.norm_stats and f"{key}_no_noops" in model_obj.norm_stats:
        key = f"{key}_no_noops"
    assert key in model_obj.norm_stats, f"unnorm_key '{key}' not found"
    cfg.unnorm_key = key

    proprio_projector = get_proprio_projector(cfg, model_obj.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model_obj.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    noisy_ap = get_noisy_action_projector(cfg, model_obj.llm_dim) if cfg.use_diffusion else None
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    resize_size = get_image_resize_size(cfg)

    # Load task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # BDDL language override
    if cfg.use_bddl_language:
        import re
        pattern = re.compile(r"\(:language\s*(.*?)\)", re.IGNORECASE | re.DOTALL)
        bddl_dir = str(Path(task_suite.get_task_bddl_file_path(cfg.task_id)).parent)
        bddl_file = Path(task_suite.get_task_bddl_file_path(cfg.task_id))
        with bddl_file.open("r") as f:
            content = f.read()
        m = pattern.findall(content)
        if m:
            task_description = m[0].strip()

    print(f"\n{'='*70}")
    print(f"Task {cfg.task_id}: {task_description}")
    print(f"Episode idx: {cfg.episode_idx}")
    print(f"Max steps to run: {cfg.max_steps}")
    print(f"{'='*70}\n")

    # Reset and set initial state
    env.reset()
    obs = env.set_init_state(initial_states[cfg.episode_idx])
    sim = env.sim
    body_map = _build_moveable_body_map(sim)
    initial_z = {name: float(sim.data.body_xpos[bid][2]) for name, bid in body_map.items()}

    print(f"Objects in scene: {list(body_map.keys())}")
    print(f"Initial z values: { {k: f'{v:.4f}' for k,v in initial_z.items()} }\n")

    # Header
    obj_names = list(body_map.keys())
    grip_col = "grip_qpos[0]"
    dist_cols = [f"dist_{n[:12]}" for n in obj_names]
    dz_cols   = [f"dz_{n[:12]}" for n in obj_names]
    header = f"{'step':>5}  {grip_col:>12}  " + \
             "  ".join(f"{c:>16}" for c in dist_cols) + "  " + \
             "  ".join(f"{c:>16}" for c in dz_cols)
    print(header)
    print("-" * len(header))

    log_lines = [header, "-" * len(header)]

    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    t = 0

    while t < cfg.max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation = _prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
            actions = get_action(
                cfg, model_obj, observation, task_description,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_ap,
                use_film=cfg.use_film,
                auto_regression=cfg.auto_regression,
            )
            action_queue.extend(actions)

        action = action_queue.popleft()
        action = normalize_gripper_action(action, binarize=True)
        if cfg.model_family == "openvla":
            action = invert_gripper_action(action)
        obs, _, done, _ = env.step(action.tolist())

        # Collect sensor values
        eef_pos = obs["robot0_eef_pos"]
        grip_val = float(obs["robot0_gripper_qpos"][0])
        dists = {}
        dzs = {}
        for name, bid in body_map.items():
            obj_pos = sim.data.body_xpos[bid]
            dists[name] = float(np.linalg.norm(eef_pos - obj_pos))
            dzs[name] = float(obj_pos[2]) - initial_z[name]

        # Format and print
        dist_str = "  ".join(f"{dists[n]:>16.4f}" for n in obj_names)
        dz_str   = "  ".join(f"{dzs[n]:>16.4f}" for n in obj_names)
        # Highlight steps where gripper is closing AND something is close
        flag = ""
        if grip_val < 0.035:
            min_dist = min(dists.values())
            max_dz   = max(dzs.values())
            if min_dist < 0.10:
                flag = "  <<< GRIP+NEAR"
            if min_dist < 0.10 and max_dz > 0.01:
                flag = "  <<< PICK EVENT"

        line = f"{t:>5}  {grip_val:>12.4f}  {dist_str}  {dz_str}{flag}"
        print(line)
        log_lines.append(line)

        if done:
            print(f"\n  [env done at step {t}]")
            break
        t += 1

    env.close()

    # Save log
    log_path = "debug_pick_thresholds.log"
    with open(log_path, "w") as f:
        f.write(f"Task: {task_description}\n\n")
        f.write("\n".join(log_lines))
    print(f"\nLog saved to {log_path}")
    print("\n--- Threshold recommendations ---")
    print("  prox_thresh : look at dist columns when '<<< PICK EVENT' appears → use that value × 1.2")
    print("  grip_thresh : look at grip_qpos[0] when picking      → use that value × 1.2")
    print("  lift_thresh : look at dz columns when picking        → use a value smaller than the observed dz")


if __name__ == "__main__":
    debug_thresholds()
