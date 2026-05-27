"""
eval_pick_rate.py

Measure how often a trained policy picks the correct object as specified by the
language instruction.

For each trial, the script:
  1. Parses the task instruction to identify the target "pick" object.
  2. Runs the rollout; detects the first pick event (EEF proximity + gripper
     closed + object lifted).
  3. Records whether the picked object matches the instructed object.

Reports per-task and aggregate pick-correct rates, saved to a JSON file.

Usage (on NV server, inside `screen exp`, after setting PYTHONPATH):
    python experiments/robot/libero/eval_pick_rate.py \
        --pretrained_checkpoint ckpt/ckpoints/<name> \
        --task_suite_name libero_spatial \
        --perturbation_mode task \
        --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml \
        --unnorm_key libero_spatial \
        --num_trials_per_task 50 \
        --use_l1_regression False \
        --use_proprio False \
        --num_images_in_input 1 \
        --task_label my_pick_rate_test

Perturbation modes:
    none   – no perturbation (standard eval)
    lan    – language-only perturbation (instruction changed, scene unchanged)
    task   – task instruction swap (instruction from a different task)
    swap   – physical object swap (objects' xy positions swapped)
    object – object replacement (different object instance used)
"""

import json
import logging
import os
import re
import sys
import yaml
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import tqdm
import draccus
import torch
from libero.libero import benchmark

from experiments.robot.libero import perturbation

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
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Max steps per suite (mirrors run_libero_pro_eval_substep.py)
# ---------------------------------------------------------------------------
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    # Perturbed variants keep the same budget
    "libero_spatial_lan": 220, "libero_spatial_task": 220,
    "libero_spatial_swap": 220, "libero_spatial_object": 220,
    "libero_spatial_temp": 220,
    "libero_object_lan": 280, "libero_object_task": 280,
    "libero_object_swap": 280, "libero_object_object": 280,
    "libero_object_temp": 280,
    "libero_goal_lan": 300, "libero_goal_task": 300,
    "libero_goal_swap": 300, "libero_goal_object": 300,
    "libero_goal_temp": 300,
    "libero_10_lan": 520, "libero_10_task": 520,
    "libero_10_swap": 520, "libero_10_object": 520,
    "libero_10_temp": 520,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PickRateConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    auto_regression: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Env
    task_suite_name: str = "libero_spatial"
    perturbation_mode: str = "none"        # none | lan | task | swap | object
    evaluation_config_path: Optional[str] = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    env_img_res: int = 256
    single_task_id: int = -1               # -1 = all tasks

    # BDDL language
    use_bddl_language: bool = False

    # Pick detection thresholds
    prox_thresh: float = 0.06             # EEF-to-object proximity (m)
    grip_thresh: float = 0.030            # gripper closed if qpos < this
    lift_thresh: float = 0.015            # object lifted above initial z by this

    # Required by get_vla_action / get_action internals (keep defaults, don't expose to user)
    remove_wrap: bool = False
    h_decoding: bool = False
    e_decoding: bool = False
    energy_alpha: float = 0.5
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50

    # Video / visualization
    save_video: bool = False           # save annotated MP4 per episode
    compute_attention: bool = False    # overlay ViT patch saliency on video frames
    video_dir: str = "./experiments/logs/videos"
    video_fps: int = 10

    # Output
    task_label: str = ""
    local_log_dir: str = "./experiments/logs"
    seed: int = 7


# ---------------------------------------------------------------------------
# Pick detection helpers
# ---------------------------------------------------------------------------

def _build_moveable_body_map(sim) -> Dict[str, int]:
    """name → body_id for all free-jointed non-robot bodies."""
    mapping = {}
    for jnt_id, jnt_type in enumerate(sim.model.jnt_type):
        if jnt_type != 0:  # 0 = mjJNT_FREE
            continue
        body_id = sim.model.jnt_bodyid[jnt_id]
        body_name = sim.model.body_id2name(body_id)
        if "robot" in body_name.lower():
            continue
        mapping[body_name] = body_id
    return mapping


def _get_initial_z(sim, body_map: Dict[str, int]) -> Dict[str, float]:
    return {name: float(sim.data.body_xpos[bid][2]) for name, bid in body_map.items()}


def _detect_picked_object(
    obs,
    sim,
    body_map: Dict[str, int],
    initial_z: Dict[str, float],
    prox_thresh: float,
    grip_thresh: float,
    lift_thresh: float,
) -> Optional[str]:
    """Return the body name of the object being picked, or None.

    Criteria (all must hold):
      1. Gripper is closed (robot0_gripper_qpos[0] < grip_thresh).
      2. EEF is within prox_thresh of the object's current position.
      3. Object has been lifted by at least lift_thresh above its initial z.
    """
    eef_pos = obs.get("robot0_eef_pos")
    grip = obs.get("robot0_gripper_qpos")
    if eef_pos is None or grip is None:
        return None

    if float(grip[0]) >= grip_thresh:
        return None  # gripper open

    best_name, best_dist = None, float("inf")
    for name, bid in body_map.items():
        obj_pos = sim.data.body_xpos[bid]
        dist = float(np.linalg.norm(eef_pos - obj_pos))
        if dist < prox_thresh and dist < best_dist:
            init_z = initial_z.get(name, 0.0)
            if float(obj_pos[2]) > init_z + lift_thresh:
                best_dist = dist
                best_name = name
    return best_name


def _parse_target_object(instruction: str, body_map: Dict[str, int]) -> Optional[str]:
    """Identify which object the instruction asks the robot to pick.

    Body names like 'akita_black_bowl_1_main' are cleaned by:
      - splitting on '_'
      - dropping pure-digit tokens and known non-semantic suffixes
    Then all contiguous sub-phrases of the cleaned tokens are tried against
    the instruction text.  The body name whose longest sub-phrase appears in
    the instruction wins.  This handles brand prefixes ("akita"), instance
    numbers ("1","2"), and suffixes ("main","base","top").
    """
    STRIP_TOKENS = {"main", "base", "top", "bottom", "left", "right"}
    instruction_lower = instruction.lower()

    # For each body, find the best (earliest, longest) phrase match in the instruction.
    # Sort key: (first_occurrence_pos, -phrase_len) — earlier and longer wins.
    # This correctly picks the FIRST object mentioned (the one to pick), not the
    # destination object which tends to appear later.
    candidates = []  # (first_pos, phrase_len, body_name)

    for name in body_map:
        tokens = name.lower().split("_")
        cleaned = [t for t in tokens if t and not t.isdigit() and t not in STRIP_TOKENS]
        if not cleaned:
            continue
        for start in range(len(cleaned)):
            for end in range(len(cleaned), start, -1):
                phrase = " ".join(cleaned[start:end])
                pos = instruction_lower.find(phrase)
                if pos != -1:
                    candidates.append((pos, len(phrase), name))

    if not candidates:
        return None
    # Earliest occurrence wins; among ties, longest phrase wins
    candidates.sort(key=lambda x: (x[0], -x[1]))
    return candidates[0][2]


# ---------------------------------------------------------------------------
# BDDL helpers (mirrors run_libero_pro_eval_substep.py)
# ---------------------------------------------------------------------------

def _extract_task_from_bddl(bddl_dir_str: str, task_suite) -> List[str]:
    pattern = re.compile(r"\(:language\s*(.*?)\)", re.IGNORECASE | re.DOTALL)
    tasks = []
    bddl_dir = Path(bddl_dir_str)
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        bddl_file = bddl_dir / task.bddl_file
        with bddl_file.open("r", encoding="utf-8") as f:
            content = f.read()
        matches = pattern.findall(content)
        if matches:
            tasks.append(matches[0].strip())
        else:
            tasks.append(content.split(":language")[1].split(")")[0].strip())
    return tasks


# ---------------------------------------------------------------------------
# Model initialisation (mirrors the existing eval script)
# ---------------------------------------------------------------------------

def _initialize_model(cfg: PickRateConfig):
    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        _check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def _check_unnorm_key(cfg: PickRateConfig, model) -> None:
    key = cfg.unnorm_key or cfg.task_suite_name
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    assert key in model.norm_stats, f"unnorm_key '{key}' not in model.norm_stats!"
    cfg.unnorm_key = key


# ---------------------------------------------------------------------------
# Perturbation config setup (mirrors run_libero_pro_eval_substep.py Step 2)
# ---------------------------------------------------------------------------

def _apply_perturbation_config(cfg: PickRateConfig) -> None:
    """Modify cfg.task_suite_name based on perturbation_mode, creating env if needed."""
    if cfg.perturbation_mode == "none" or not cfg.evaluation_config_path:
        return

    mode = cfg.perturbation_mode
    assert mode in ("lan", "task", "swap", "object"), \
        f"perturbation_mode must be one of: none, lan, task, swap, object. Got: {mode}"

    with open(cfg.evaluation_config_path, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    # Set only the requested flag
    for flag in ("use_language", "use_task", "use_swap", "use_object", "use_environment"):
        eval_cfg[flag] = False
    flag_key = {"lan": "use_language", "task": "use_task",
                "swap": "use_swap", "object": "use_object"}[mode]
    eval_cfg[flag_key] = True

    eval_cfg["bddl_files_path"] = eval_cfg.get("bddl_files_path", "") + "/" + cfg.task_suite_name
    eval_cfg["task_suite_name"] = cfg.task_suite_name

    perturb_suffix = eval_cfg.get("perturbation_mapping", {}).get(flag_key, mode)
    init_file_path = eval_cfg.get("init_file_dir", "") + cfg.task_suite_name + "_" + perturb_suffix

    if not os.path.exists(init_file_path):
        perturbation.create_env(configs=eval_cfg)

    cfg.task_suite_name = cfg.task_suite_name + "_" + perturb_suffix
    logger.info(f"Perturbation applied: task_suite_name → {cfg.task_suite_name}")


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_r = resize_image_for_policy(img, resize_size)
    wrist_r = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_r,
        "wrist_image": wrist_r,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation


def _process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


# ---------------------------------------------------------------------------
# Attention / saliency helpers
# ---------------------------------------------------------------------------

def _extract_patch_saliency(model):
    """Capture ViT attention weights via register_forward_hook on blocks[-2].attn.

    The hook fires after the attention module's forward (which may use fused/flash
    attention that doesn't expose weights).  Inside the hook we receive the module's
    INPUT and recompute attention weights manually in torch.no_grad() — this avoids
    any monkey-patching issues while still getting the correct attention map.

    Method — "attention received" per patch (no CLS models like SigLIP):
        received_j = mean over heads and query positions of attn[:, :, :, j]
    Semantically meaningful regions (objects) receive consistently higher attention
    from surrounding patches than plain background.
    """
    class _AttnCtx:
        def __init__(self):
            self._cache = [None]
            self._handle = None
            self._fallback = False
            cache = self._cache
            try:
                am = model.vision_backbone.featurizer.blocks[-2].attn
                logger.warning(f"[ATTN] registering hook on {type(am).__name__} (blocks[-2].attn)")

                def _hook(module, inp, output):
                    # inp[0]: [B, N, C] — patch tokens fed into attention
                    # Keep original dtype (bf16) to match module weights; cast to float32 only at end
                    try:
                        x = inp[0].detach()          # keep bf16 — matches qkv.weight dtype
                        B, N, C = x.shape
                        num_heads = module.num_heads
                        head_dim = C // num_heads
                        scale = head_dim ** -0.5

                        with torch.no_grad():
                            qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                            q, k, _ = qkv.unbind(0)
                            if getattr(module, 'q_norm', None) is not None:
                                q = module.q_norm(q)
                            if getattr(module, 'k_norm', None) is not None:
                                k = module.k_norm(k)
                            attn_w = (q.float() * scale) @ k.float().transpose(-2, -1)  # cast here
                            attn_w = attn_w.softmax(dim=-1)
                        cache[0] = attn_w.cpu()      # already float32 from the cast above
                        logger.warning(f"[ATTN hook] captured shape={tuple(attn_w.shape)}")
                    except Exception as e:
                        logger.warning(f"[ATTN hook] error: {type(e).__name__}: {e}")

                self._handle = am.register_forward_hook(_hook)
            except (AttributeError, IndexError) as e:
                logger.warning(f"[ATTN] blocks[-2].attn not found ({e}); falling back to L2-norm")
                self._fallback = True
                try:
                    def _norm_hook(module, inp, output):
                        try:
                            feats = output.detach().float()  # [1, N, D]
                            norms = feats[0].norm(dim=-1)    # [N]
                            cache[0] = norms                 # 1-D tensor
                        except Exception as e2:
                            logger.warning(f"[ATTN norm hook] {e2}")
                    self._handle = model.vision_backbone.register_forward_hook(_norm_hook)
                except Exception as e2:
                    logger.warning(f"[ATTN] fallback hook also failed: {e2}")
                    self._handle = None

        def get(self) -> Optional[np.ndarray]:
            if self._handle is not None:
                self._handle.remove()
            if self._cache[0] is None:
                logger.warning("[ATTN] cache is None — hook did not fire")
                return None
            try:
                data = self._cache[0]
                if self._fallback:
                    # L2-norm fallback: data is 1-D [N]
                    norms = data.cpu().numpy()
                    N = norms.shape[0]
                    side = int(round(N ** 0.5))
                    if side * side != N:
                        return None
                    sal = norms.reshape(side, side)
                else:
                    # Attention path: data is [1, heads, N, N]
                    attn_avg = data[0].mean(dim=0).cpu().numpy()  # [N, N]
                    N = attn_avg.shape[0]

                    # Find the token layout by trying common ViT token structures:
                    #   pure spatial:      N = side²            (SigLIP, no CLS)
                    #   CLS + spatial:     N = 1 + side²        (standard ViT)
                    #   CLS + regs + spatial: N = 1 + R + side² (DINOv2-with-registers, R=4 typical)
                    # For attention map we use CLS→spatial attention (row 0, skipping non-spatial tokens).
                    spatial_start = None
                    side = None
                    for num_prefix in [0, 1, 2, 4, 5, 8]:   # 0=no prefix, 1=CLS, 1+4=CLS+4regs, etc.
                        n_spatial = N - num_prefix
                        if n_spatial <= 0:
                            continue
                        s = int(round(n_spatial ** 0.5))
                        if s * s == n_spatial:
                            spatial_start = num_prefix
                            side = s
                            break

                    if spatial_start is None:
                        logger.warning(f"[ATTN] Cannot parse N={N} into a spatial grid; skip")
                        return None

                    logger.warning(f"[ATTN] N={N}: prefix={spatial_start} + {side}×{side} spatial patches")

                    if spatial_start == 0:
                        # No CLS: "attention received" — how much each patch is attended to
                        received = attn_avg.mean(axis=0)   # [N]
                    else:
                        # CLS→spatial: which patches does the CLS token attend to?
                        received = attn_avg[0, spatial_start:]  # [side²]

                    sal = received.reshape(side, side)

                sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
                return sal
            except Exception as e:
                logger.debug(f"[ATTN] get() error: {e}")
                return None

    return _AttnCtx()


def _blend_saliency_on_frame(img_rgb: np.ndarray, sal: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    """Blend a [H_grid, W_grid] saliency map (jet colormap) onto an RGB image."""
    H, W = img_rgb.shape[:2]
    sal_u8 = (sal * 255).astype(np.uint8)
    sal_resized = cv2.resize(sal_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(sal_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = img_rgb.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _wrap_text_px(text: str, font, fs: float, th: int, max_px: int) -> List[str]:
    """Word-wrap text so each line fits within max_px pixels (measured with cv2.getTextSize)."""
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []
    for word in words:
        candidate = " ".join(cur + [word])
        (w, _), _ = cv2.getTextSize(candidate, font, fs, th)
        if w <= max_px:
            cur.append(word)
        else:
            if cur:
                lines.append(" ".join(cur))
                cur = [word]
            else:
                # Single word wider than max — truncate character-by-character
                trunc = word
                while trunc:
                    (w, _), _ = cv2.getTextSize(trunc, font, fs, th)
                    if w <= max_px:
                        break
                    trunc = trunc[:-1]
                lines.append(trunc or word[:4])
                cur = []
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


# Pre-compute a fixed BAR_HEIGHT once so every call to _annotate_frame
# produces the same canvas height (required for MP4 streams).
# We define N_LINES=5, and compute lh from a reference font scale.
_ANNOT_N_LINES = 5


def _annotate_frame(
    img_rgb: np.ndarray,
    instruction: str,
    expected_obj: Optional[str],
    picked_obj: Optional[str],
    step: int,
    pick_correct: Optional[bool],
) -> np.ndarray:
    """Add a fixed-height info bar ABOVE the image.

    Always renders exactly 5 lines regardless of content, so all frames in a video
    have the same height.  Total height is padded to be divisible by 16 (avoids
    ffmpeg macro-block warnings).
    """
    H, W = img_rgb.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad  = 4
    th   = 1

    # Font scale: choose so a typical 30-char line fits in the image width
    # Start from a target, then reduce until "M"*30 fits.
    fs = 0.45
    ref_text = "M" * 28
    while fs > 0.20:
        (tw, _), _ = cv2.getTextSize(ref_text, font, fs, th)
        if tw <= W - 2 * pad:
            break
        fs -= 0.02
    fs = round(fs, 2)

    lh = max(14, int(cv2.getTextSize("Ag", font, fs, th)[0][1] * 2.2))  # line height

    max_px = W - 2 * pad

    # Instruction: wrap to at most 2 lines (pixel-accurate)
    instr_lines = _wrap_text_px(instruction, font, fs, th, max_px)[:2]
    while len(instr_lines) < 2:
        instr_lines.append("")

    # Picked line
    if picked_obj is not None:
        status_str = "CORRECT" if pick_correct else "WRONG"
        pick_color = (100, 240, 100) if pick_correct else (255, 100, 100)
        pick_text = f"Pick:{picked_obj[:22]} [{status_str}]"
    else:
        pick_color = (140, 140, 140)
        pick_text = "Pick: ---"

    # Fixed 5-line layout
    text_lines = [
        (instr_lines[0],                           (220, 220, 220)),
        (instr_lines[1],                           (195, 195, 195)),
        ("Exp: " + (expected_obj or "?")[:30],     (140, 205, 255)),
        (pick_text,                                 pick_color),
        (f"Step {step}",                            (120, 120, 120)),
    ]
    assert len(text_lines) == _ANNOT_N_LINES

    bar_h = pad + lh * _ANNOT_N_LINES + pad

    # Pad total height to multiple of 16 (avoids ffmpeg macro-block resize)
    total_h = bar_h + H
    padded_h = ((total_h + 15) // 16) * 16
    extra = padded_h - total_h        # black rows added at bottom of image

    canvas = np.zeros((padded_h, W, 3), dtype=np.uint8)
    canvas[bar_h: bar_h + H, :] = img_rgb  # image sits after bar; bottom extra rows stay black

    for i, (text, color) in enumerate(text_lines):
        if not text:
            continue
        y = pad + lh * i + lh - 3
        cv2.putText(canvas, text, (pad, y), font, fs, color, th, cv2.LINE_AA)

    return canvas


def _save_episode_video(
    frames: List[np.ndarray],
    video_path: str,
    fps: int = 10,
) -> None:
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode_pick(
    cfg: PickRateConfig,
    env,
    task_description: str,
    expected_object: Optional[str],
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state,
    trial_label: str = "",
) -> Tuple[Optional[str], bool]:
    """Run one episode until a pick is detected, then immediately return.

    Returns:
        picked_object – body name of first object picked (None if never picked)
        pick_correct  – True if picked_object matches expected_object
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    sim = env.sim
    body_map = _build_moveable_body_map(sim)
    initial_z = _get_initial_z(sim, body_map)

    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 300)
    t = 0
    picked_object = None

    # Per-step state for video annotation
    video_frames: List[np.ndarray] = []
    current_saliency: Optional[np.ndarray] = None  # reused across action chunk steps

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        observation = _prepare_observation(obs, resize_size)

        if len(action_queue) == 0:
            # Optionally register saliency hook BEFORE get_action
            sal_ctx = _extract_patch_saliency(model) if cfg.compute_attention else None

            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
                auto_regression=cfg.auto_regression,
            )
            action_queue.extend(actions)

            # Retrieve saliency captured during get_action forward pass
            if sal_ctx is not None:
                current_saliency = sal_ctx.get()

        action = _process_action(action_queue.popleft(), cfg.model_family)
        obs, _, done, _ = env.step(action.tolist())

        # Build annotated frame for video
        if cfg.save_video:
            raw_img = get_libero_image(obs)          # HxWx3 RGB, already rotated
            frame = raw_img.copy()
            if cfg.compute_attention and current_saliency is not None:
                frame = _blend_saliency_on_frame(frame, current_saliency)
            frame = _annotate_frame(
                frame, task_description, expected_object,
                picked_object, t,
                (picked_object == expected_object) if picked_object else None,
            )
            video_frames.append(frame)

        picked_object = _detect_picked_object(
            obs, sim, body_map, initial_z,
            cfg.prox_thresh, cfg.grip_thresh, cfg.lift_thresh,
        )
        if picked_object is not None:
            # Annotate final frame with pick result
            if cfg.save_video:
                raw_img = get_libero_image(obs)
                frame = raw_img.copy()
                if cfg.compute_attention and current_saliency is not None:
                    frame = _blend_saliency_on_frame(frame, current_saliency)
                frame = _annotate_frame(
                    frame, task_description, expected_object,
                    picked_object, t,
                    picked_object == expected_object if expected_object else None,
                )
                video_frames.append(frame)
            break

        if done:
            break
        t += 1

    else:
        picked_object = None

    pick_correct = (
        picked_object is not None
        and expected_object is not None
        and picked_object == expected_object
    )

    # Save video
    if cfg.save_video and video_frames:
        correct_str = "correct" if pick_correct else ("none" if picked_object is None else "wrong")
        safe_label = re.sub(r"[^\w\-]", "_", trial_label)[:60]
        video_path = os.path.join(cfg.video_dir, f"{safe_label}_{correct_str}.mp4")
        _save_episode_video(video_frames, video_path, fps=cfg.video_fps)
        logger.info(f"  Video saved: {video_path}")

    return picked_object, pick_correct


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@draccus.wrap()
def eval_pick_rate(cfg: PickRateConfig) -> None:
    assert cfg.pretrained_checkpoint, "--pretrained_checkpoint is required"

    set_seed_everywhere(cfg.seed)

    # Apply perturbation (modifies cfg.task_suite_name in-place)
    _apply_perturbation_config(cfg)

    # Load model
    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Set up benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # BDDL language override
    bddl_descriptions = None
    if cfg.use_bddl_language:
        bddl_dir = str(Path(task_suite.get_task_bddl_file_path(0)).parent)
        bddl_descriptions = _extract_task_from_bddl(bddl_dir, task_suite)
        assert len(bddl_descriptions) == num_tasks

    # Logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    label = cfg.task_label or f"pick_rate_{cfg.task_suite_name}_{DATE_TIME}"
    log_path = os.path.join(cfg.local_log_dir, f"{label}.txt")
    log_file = open(log_path, "w")

    def log(msg):
        logger.info(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # num_trials_per_task is repurposed as total_trials here
    total_trials = cfg.num_trials_per_task

    log(f"Checkpoint   : {cfg.pretrained_checkpoint}")
    log(f"Task suite   : {cfg.task_suite_name}")
    log(f"Perturbation : {cfg.perturbation_mode}")
    log(f"Total trials : {total_trials}  (cycled across {num_tasks} tasks)")

    # Pre-load all task metadata (descriptions, body maps, envs) once
    task_ids = [cfg.single_task_id] if cfg.single_task_id >= 0 else list(range(num_tasks))
    tasks_meta = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        if bddl_descriptions:
            task_description = bddl_descriptions[task_id]
        env.reset()
        body_map = _build_moveable_body_map(env.sim)
        expected_object = _parse_target_object(task_description, body_map)
        log(f"  Task {task_id}: {task_description}")
        log(f"    Expected: {expected_object}  bodies: {list(body_map.keys())}")
        tasks_meta.append({
            "task_id": task_id,
            "task_description": task_description,
            "expected_object": expected_object,
            "initial_states": initial_states,
            "env": env,
            "picks": 0,
            "correct_picks": 0,
            "episodes": 0,
        })

    # Distribute total_trials across tasks (round-robin)
    total_episodes = total_picks = total_correct_picks = 0

    for trial_idx in tqdm.tqdm(range(total_trials)):
        meta = tasks_meta[trial_idx % len(tasks_meta)]
        ep_idx = trial_idx // len(tasks_meta)  # which initial state to use
        initial_state = meta["initial_states"][ep_idx % len(meta["initial_states"])]

        trial_label = f"{label}_trial{trial_idx:03d}_task{meta['task_id']}"
        picked_obj, pick_correct = _run_episode_pick(
            cfg, meta["env"], meta["task_description"], meta["expected_object"],
            model, resize_size, processor,
            action_head, proprio_projector, noisy_action_projector,
            initial_state,
            trial_label=trial_label,
        )

        meta["episodes"] += 1
        if picked_obj is not None:
            meta["picks"] += 1
        if pick_correct:
            meta["correct_picks"] += 1

        total_episodes += 1
        if picked_obj is not None:
            total_picks += 1
        if pick_correct:
            total_correct_picks += 1

        log(
            f"  trial {trial_idx:3d} | task={meta['task_id']} | "
            f"picked={picked_obj} | expected={meta['expected_object']} | correct={pick_correct}"
        )

    # Close all envs
    for meta in tasks_meta:
        meta["env"].close()

    # Per-task summary
    all_results = []
    for meta in tasks_meta:
        n = meta["episodes"]
        pick_rate = meta["correct_picks"] / n if n > 0 else 0.0
        pick_any = meta["picks"] / n if n > 0 else 0.0
        log(
            f"  Task {meta['task_id']} ({meta['episodes']} trials) | "
            f"pick-any: {pick_any:.2f} | pick-correct: {pick_rate:.2f}"
        )
        all_results.append({
            "task_id": meta["task_id"],
            "task_description": meta["task_description"],
            "expected_object": meta["expected_object"],
            "episodes": n,
            "picks": meta["picks"],
            "correct_picks": meta["correct_picks"],
            "pick_any_rate": pick_any,
            "pick_correct_rate": pick_rate,
        })

    # Final summary
    overall_pick_any = total_picks / total_episodes if total_episodes > 0 else 0.0
    overall_pick_correct = total_correct_picks / total_episodes if total_episodes > 0 else 0.0

    log("\n========== FINAL RESULTS ==========")
    log(f"Total trials     : {total_episodes}")
    log(f"Pick-any rate    : {overall_pick_any:.4f} ({overall_pick_any*100:.1f}%)")
    log(f"Pick-correct rate: {overall_pick_correct:.4f} ({overall_pick_correct*100:.1f}%)")

    # Save JSON results
    results_dict = {
        "checkpoint": str(cfg.pretrained_checkpoint),
        "task_suite_name": cfg.task_suite_name,
        "perturbation_mode": cfg.perturbation_mode,
        "total_trials": total_trials,
        "overall": {
            "pick_any_rate": overall_pick_any,
            "pick_correct_rate": overall_pick_correct,
        },
        "per_task": all_results,
    }

    json_path = os.path.join(cfg.local_log_dir, f"{label}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    log(f"\nResults saved to: {json_path}")
    log(f"Log saved to    : {log_path}")

    log_file.close()


if __name__ == "__main__":
    eval_pick_rate()
