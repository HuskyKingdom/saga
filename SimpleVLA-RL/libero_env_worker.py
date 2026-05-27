"""
Standalone LIBERO environment worker for multiprocessing with spawn start method.

This module intentionally imports ONLY lightweight dependencies (libero, numpy, gc)
so that spawned subprocesses do not inherit or re-import heavy libraries such as
transformers, tensorflow, or PyTorch.

Task metadata (bddl_file, task_description, initial_state) are pre-loaded by the
parent process and passed as arguments, so this worker never needs to import torch
or libero.benchmark.
"""
import gc
import os
import random as _random
import sys
import traceback
import numpy as np


def _get_libero_dummy_action(model_family: str):
    return [0, 0, 0, 0, 0, 0, -1]


def _parse_obj_of_interest(bddl_file: str) -> list:
    """Return the list of object names listed under (:obj_of_interest ...) in the BDDL file.

    These are the task-relevant objects (targets + goal containers).
    Falls back to an empty list if the file cannot be parsed.
    """
    try:
        with open(bddl_file, "r") as f:
            text = f.read()
        start = text.find("(:obj_of_interest")
        if start == -1:
            return []
        end = text.find(")", start)
        block = text[start + len("(:obj_of_interest"):end]
        return [tok.strip() for tok in block.split() if tok.strip()]
    except Exception:
        return []


def _random_swap_objects(sim, seed=None, max_distance=1e9, bddl_file=None):
    """Randomly swap the (x, y) table positions of two moveable objects.

    Only x and y are swapped; z is kept so objects remain at their original
    height and don't fall through the table or become unreachable.

    Free joints (mjJNT_FREE = 0) own a 7-value qpos block: [x, y, z, qw, qx, qy, qz].
    Robot bodies are excluded by name prefix.

    Parameters
    ----------
    sim          : MjSim
    seed         : int or None
        When provided, seeds the RNG before sampling so that all rollouts that
        share the same (task, trial) — i.e. the GRPO n_samples group — pick the
        same swap pair and therefore start from identical visual observations.
    max_distance : float
        Curriculum gate (metres).  Only pairs whose current (x, y) separation
        is <= max_distance are eligible.  Start small and grow over training to
        implement a swap-distance curriculum.
    """
    rng = _random.Random(seed)  # isolated RNG, does not affect global state

    # Parse which bodies are task-relevant (obj_of_interest in BDDL)
    obj_of_interest = set(_parse_obj_of_interest(bddl_file)) if bddl_file else set()

    swappable = []
    for jnt_id, jnt_type in enumerate(sim.model.jnt_type):
        if jnt_type != 0:  # 0 = mjJNT_FREE (6-DOF)
            continue
        body_id = sim.model.jnt_bodyid[jnt_id]
        body_name = sim.model.body_id2name(body_id)
        if "robot" in body_name.lower():
            continue
        qpos_addr = sim.model.jnt_qposadr[jnt_id]
        swappable.append((body_name, qpos_addr))

    if len(swappable) < 2:
        return

    def _within_distance(qa, qb):
        dx = sim.data.qpos[qa]     - sim.data.qpos[qb]
        dy = sim.data.qpos[qa + 1] - sim.data.qpos[qb + 1]
        return (dx * dx + dy * dy) ** 0.5 <= max_distance

    # Preferred strategy: swap one obj_of_interest with one distractor.
    # This ensures a task-relevant object is always displaced, forcing the
    # model to rely on language rather than memorised positions.
    # For multi-target tasks (e.g. libero-10 "pick both A and B"), this
    # displaces exactly one target per episode, keeping partial structure intact.
    targets = [(n, a) for n, a in swappable if n in obj_of_interest]
    others  = [(n, a) for n, a in swappable if n not in obj_of_interest]

    eligible = [
        (t, o) for t in targets for o in others
        if _within_distance(t[1], o[1])
    ]

    if not eligible:
        # Fallback: any pair within distance (obj_of_interest info unavailable
        # or all bodies are obj_of_interest — rare but handled gracefully)
        eligible = [
            (swappable[i], swappable[j])
            for i in range(len(swappable))
            for j in range(i + 1, len(swappable))
            if _within_distance(swappable[i][1], swappable[j][1])
        ]

    if not eligible:
        return  # no pair within current curriculum distance — skip swap

    (_, qa), (_, qb) = rng.choice(eligible)

    # Swap x and y; preserve z and orientation
    x_a, y_a = float(sim.data.qpos[qa]),     float(sim.data.qpos[qa + 1])
    x_b, y_b = float(sim.data.qpos[qb]),     float(sim.data.qpos[qb + 1])
    sim.data.qpos[qa],     sim.data.qpos[qa + 1] = x_b, y_b
    sim.data.qpos[qb],     sim.data.qpos[qb + 1] = x_a, y_a

    sim.forward()  # propagate new positions to derived quantities


def _get_target_body_ids(sim, bddl_file):
    """Return MuJoCo body IDs for obj_of_interest objects.

    Matches by exact free-joint body name (same logic as _random_swap_objects).
    Falls back to all free-jointed non-robot bodies if obj_of_interest is empty.
    """
    obj_names = set(_parse_obj_of_interest(bddl_file)) if bddl_file else set()
    target_ids = []
    fallback_ids = []
    for jnt_id, jnt_type in enumerate(sim.model.jnt_type):
        if jnt_type != 0:
            continue
        body_id = sim.model.jnt_bodyid[jnt_id]
        body_name = sim.model.body_id2name(body_id)
        if "robot" in body_name.lower():
            continue
        fallback_ids.append(body_id)
        if body_name in obj_names:
            target_ids.append(body_id)
    return target_ids if target_ids else fallback_ids


def _min_dist_to_targets(obs, sim, target_body_ids):
    """Return the minimum Euclidean distance from the EEF to any target body.

    Returns float('inf') if EEF position is unavailable.
    """
    eef_pos = obs.get("robot0_eef_pos")
    if eef_pos is None or not target_body_ids:
        return float("inf")
    min_d = float("inf")
    for bid in target_body_ids:
        d = float(np.linalg.norm(eef_pos - sim.data.body_xpos[bid]))
        if d < min_d:
            min_d = d
    return min_d


# ---------------------------------------------------------------------------
# SAGA: Object-Aware Substep Tracking (lightweight, no torch dependency)
# ---------------------------------------------------------------------------

def _build_moveable_body_map(sim):
    """Build name→body_id mapping for all free-jointed non-robot bodies."""
    mapping = {}
    for jnt_id, jnt_type in enumerate(sim.model.jnt_type):
        if jnt_type != 0:  # mjJNT_FREE = 0
            continue
        body_id = sim.model.jnt_bodyid[jnt_id]
        body_name = sim.model.body_id2name(body_id)
        if "robot" in body_name.lower():
            continue
        mapping[body_name] = body_id
    return mapping


def _resolve_saga_substep_config(sim, key_steps, bddl_file):
    """Resolve target body IDs for SAGA substep config.

    Args:
        sim: MjSim instance (after env.reset and optional swap).
        key_steps: list of {"type": "pick"/"place", "subgoal": str}
        bddl_file: path to BDDL file (for obj_of_interest fallback).

    Returns:
        list of {"type", "subgoal", "target_object", "target_body_id"} or None on failure.
    """
    body_map = _build_moveable_body_map(sim)
    if not body_map:
        return None

    config = []
    for step in key_steps:
        subgoal_lower = step["subgoal"].lower()
        target_name = None
        target_bid = None
        # Match body name (underscores→spaces) against subgoal text
        for bname, bid in body_map.items():
            natural = bname.replace("_", " ")
            if natural in subgoal_lower:
                target_name = bname
                target_bid = bid
                break
        config.append({
            "type": step["type"],
            "subgoal": step["subgoal"],
            "target_object": target_name,
            "target_body_id": target_bid,
        })
    return config


class SagaSubstepTracker:
    """Object-aware substep tracker for SAGA.

    Tracks pick/place substep completion during RL rollouts.
    Pick detection is conditioned on the subgoal-specified target object.
    Place detection uses the environment's native success signal.

    Uses only numpy (no torch) to stay lightweight in spawned subprocesses.
    """

    def __init__(self, sim, substep_config, env,
                 prox_thresh=0.05, grip_thresh=0.025, lift_thresh=0.02):
        self.sim = sim
        self.env = env
        self.substep_config = substep_config
        self.n_substeps = len(substep_config)
        self.prox_thresh = prox_thresh
        self.grip_thresh = grip_thresh
        self.lift_thresh = lift_thresh

        # Record initial z of each target for lift detection
        self.initial_obj_z = {}
        for k, cfg in enumerate(substep_config):
            bid = cfg.get("target_body_id")
            if bid is not None:
                self.initial_obj_z[k] = float(sim.data.body_xpos[bid][2])

        # Monotonic state
        self.current_substep = 0
        self.substep_completed = [False] * self.n_substeps
        self.boundary_steps = [-1] * self.n_substeps

    def step(self, obs, env_step):
        """Called after each env.step(). Updates substep tracking.

        Args:
            obs: environment observation dict (has robot0_eef_pos, robot0_gripper_qpos).
            env_step: current finish_step count (individual env steps).
        """
        if self.current_substep >= self.n_substeps:
            return

        cfg = self.substep_config[self.current_substep]

        if cfg["type"] == "pick" and self._check_pick(obs, cfg):
            self.substep_completed[self.current_substep] = True
            self.boundary_steps[self.current_substep] = env_step
            self.current_substep += 1
        elif cfg["type"] == "place" and self._check_place():
            self.substep_completed[self.current_substep] = True
            self.boundary_steps[self.current_substep] = env_step
            self.current_substep += 1

    def _check_pick(self, obs, cfg):
        """Object-aware pick detection: only checks the subgoal target."""
        bid = cfg.get("target_body_id")
        if bid is None:
            return False

        eef_pos = obs.get("robot0_eef_pos")
        if eef_pos is None:
            return False

        obj_pos = self.sim.data.body_xpos[bid]

        # 1) Proximity: EEF close to target object
        if float(np.linalg.norm(eef_pos - obj_pos)) >= self.prox_thresh:
            return False

        # 2) Gripper closed
        grip = obs.get("robot0_gripper_qpos")
        if grip is None or float(grip[0]) >= self.grip_thresh:
            return False

        # 3) Object lifted above its initial z
        init_z = self.initial_obj_z.get(self.current_substep, 0.0)
        if float(obj_pos[2]) <= init_z + self.lift_thresh:
            return False

        return True

    def _check_place(self):
        """Place detection via environment's native success signal."""
        try:
            return self.env._check_success()
        except Exception:
            return False

    def get_results(self):
        """Return substep rewards and boundary steps as plain Python lists."""
        rewards = [1.0 if c else 0.0 for c in self.substep_completed]
        return rewards, list(self.boundary_steps)


def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0
    return inverted_action


# ---------------------------------------------------------------------------
# env_worker: the target function for each spawned LIBERO subprocess
# ---------------------------------------------------------------------------

def env_worker(task_bddl_file, task_description, initial_state,
               task_name, task_id, trial_id, config,
               input_queue, output_queue, is_valid, global_steps, max_steps,
               do_swap=False, swap_seed=None, swap_max_distance=1e9,
               saga_key_steps=None):
    """Worker process for Libero environments (spawn-safe, no torch needed).

    Parameters
    ----------
    task_bddl_file    : str   – absolute path to the BDDL file (pre-computed by parent)
    task_description  : str   – language instruction (pre-computed by parent)
    initial_state     : array – initial robot/object state (pre-loaded by parent via torch.load)
    task_name         : str   – suite name (for task_file_name logging)
    task_id           : int
    trial_id          : int
    """
    # Force robosuite EGL contexts onto logical device 0 under Ray's per-actor
    # CUDA_VISIBLE_DEVICES masking.
    #
    # robosuite has TWO checks that interact badly:
    #   A. Module-level (binding_utils.py): assert MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
    #      — substring check that fires at import time. With CUDA_VISIBLE_DEVICES='6'
    #      (Ray's per-actor mask), MUJOCO_EGL_DEVICE_ID='0' fails ('0' not in '6').
    #      Default empty MUJOCO_EGL_DEVICE_ID passes ('' in 'anything' = True).
    #   B. Runtime (egl_context.py:create_initialized_egl_device_display): reads
    #      os.environ['MUJOCO_EGL_DEVICE_ID'] first, falls back to its arg. If
    #      MjRenderContextOffscreen.__init__ wrote '6' there, our device_id=0 is
    #      silently overridden to 6, EGL queries the masked-down 1 device and
    #      raises "between 0 and 0, got 6".
    #
    # Fix: do NOT set MUJOCO_EGL_DEVICE_ID before importing robosuite (let
    # check A pass with empty value), then monkey-patch the runtime factory so
    # check B reads '0' instead of robosuite's '6'.
    import robosuite.renderers.context.egl_context as _egl_ctx_mod
    _orig_create_egl = _egl_ctx_mod.create_initialized_egl_device_display
    def _patched_create_egl(device_id=None):
        os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
        return _orig_create_egl(device_id=0)
    _egl_ctx_mod.create_initialized_egl_device_display = _patched_create_egl

    from libero.libero.envs import OffScreenRenderEnv

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    env = None
    _MAX_RETRY = 5
    for _attempt in range(_MAX_RETRY):
        try:
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
            break
        except Exception as _exc:
            # Print the real exception class + message + traceback. Without this
            # the EGL/robosuite root cause is hidden and the loop just spins.
            print(f"*** env initialization failed (attempt {_attempt+1}/{_MAX_RETRY}): "
                  f"{type(_exc).__name__}: {_exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            if env is not None:
                try:
                    env.close()
                except Exception as e:
                    print(f"error when close the env: {e}")
            gc.collect()
            print("gc collect finish")
    else:
        # All retries exhausted — propagate so Ray surfaces the failure
        # instead of silently busy-spinning forever.
        raise RuntimeError(
            f"OffScreenRenderEnv failed to initialize after {_MAX_RETRY} attempts "
            f"(BDDL: {task_bddl_file}). See traceback above."
        )

    env.reset()
    obs = env.set_init_state(initial_state)

    if do_swap:
        _random_swap_objects(env.sim, seed=swap_seed, max_distance=swap_max_distance,
                             bddl_file=task_bddl_file)

    # Pre-compute target body IDs once for distance tracking
    target_body_ids = _get_target_body_ids(env.sim, task_bddl_file)

    # SAGA: initialise object-aware substep tracker (if plan steps provided)
    saga_tracker = None
    if saga_key_steps:
        saga_config = _resolve_saga_substep_config(env.sim, saga_key_steps, task_bddl_file)
        if saga_config and any(c.get("target_body_id") is not None for c in saga_config):
            saga_tracker = SagaSubstepTracker(env.sim, saga_config, env)

    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, _, _ = env.step(_get_libero_dummy_action(config.model_family))
        t += 1

    if is_valid:
        img = obs["agentview_image"][::-1, ::-1]
        valid_images.append(img)

    output_queue.put({
        'type': 'init',
        'obs': obs,
        'task_description': task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0,
        'min_dist': float('inf'),
    })

    active = True
    complete = False
    finish_step = 0
    min_dist = float('inf')   # running minimum distance over the trajectory

    while True:
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break

        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = _normalize_gripper_action(a, binarize=True)
            inverted_action = _invert_gripper_action(normalized_action)
            obs, reward, done, info = env.step(inverted_action.tolist())

            # Track minimum gripper-to-target distance across the trajectory.
            # Multi-target tasks: use the nearest target at each step.
            d = _min_dist_to_targets(obs, env.sim, target_body_ids)
            if d < min_dist:
                min_dist = d

            # SAGA: update substep tracker after each env step
            if saga_tracker is not None:
                saga_tracker.step(obs, finish_step)

            if is_valid:
                img = obs["agentview_image"][::-1, ::-1]
                step_images.append(img)

            finish_step += 1
            if done or finish_step >= max_steps:
                active = False
                complete = done
                # Success guarantees the robot reached the object;
                # pad remaining (unseen) steps as dist=0 per the curriculum spec.
                if complete:
                    min_dist = 0.0
                break

        step_result = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            'min_dist': min_dist,
            'valid_images': step_images.copy() if is_valid else [],
        }

        # SAGA: attach substep tracking results to every step message
        if saga_tracker is not None:
            sr, sb = saga_tracker.get_results()
            step_result['saga_substep_rewards'] = sr
            step_result['saga_substep_boundary_steps'] = sb

        output_queue.put(step_result)
