"""
SAGA: Subgoal-Aware Grounded Advantage for VLA Reinforcement Learning.

Provides per-substep, object-aware advantage estimation for GRPO.
This module is fully self-contained for clean decoupling from the base GRPO pipeline.

Three stages:
  1. Offline: load APD plans → extract key substeps (pick/place) per task instruction.
  2. Online:  ObjectAwareSubstepTracker in env_worker detects substep completion.
  3. Training: compute_saga_grpo_outcome_advantage replaces trajectory-level GRPO advantage
               with per-substep, object-aware advantages.

Usage:
  - Set algorithm.adv_estimator=saga in Hydra config.
  - Set saga.enabled=True and saga.apd_plans_path to the APD_plans_scaled.json path.
  - The rollout worker loads plan configs and passes them to env workers.
  - The advantage computation in ray_trainer routes to this module.
"""

import json
import torch
from collections import defaultdict


# ---------------------------------------------------------------------------
# Stage 1: Offline plan config extraction
# ---------------------------------------------------------------------------

def load_saga_plan_configs(apd_plans_path):
    """Load APD plans and extract key substeps (pick/place) indexed by instruction.

    Args:
        apd_plans_path: Path to APD_plans_scaled.json

    Returns:
        dict: instruction_string -> list of key step dicts
              e.g. [{"type": "pick", "subgoal": "Grasp the black bowl"},
                    {"type": "place", "subgoal": "Release the bowl onto the plate"}]
    """
    with open(apd_plans_path, "r") as f:
        plan_data = json.load(f)

    configs = {}
    for entry in plan_data:
        instruction = entry["instruction"]["raw"]
        plan = entry["instruction"]["plan"]
        key_steps = [
            {"type": step["action_type"], "subgoal": step["subgoal"]}
            for step in plan
            if step["action_type"] in ("pick", "place")
        ]
        if key_steps:
            configs[instruction] = key_steps

    return configs


# ---------------------------------------------------------------------------
# Stage 3: Per-substep GRPO advantage computation
# ---------------------------------------------------------------------------

def compute_saga_grpo_outcome_advantage(
    token_level_rewards,
    eos_mask,
    index,
    substep_rewards,
    substep_boundary_steps,
    action_token_len,
    epsilon=1e-6,
):
    """Compute SAGA per-substep advantages for GRPO.

    Instead of a single scalar advantage per trajectory, SAGA computes
    independent advantages for each substep (pick, place) and assigns
    each token the advantage of its containing substep.

    For tasks without valid SAGA data (substep_rewards identical across
    substeps), this naturally degrades to standard GRPO.

    Args:
        token_level_rewards: (batch_size, response_length) — unused by SAGA
            directly, kept for API compatibility with GRPO.
        eos_mask: (batch_size, response_length) — valid token mask.
        index: array-like (batch_size,) — group uid for GRPO grouping.
        substep_rewards: (batch_size, K) — binary rewards per substep.
        substep_boundary_steps: (batch_size, K) — env step where each substep
            completed; -1 if not completed.
        action_token_len: int — tokens per env step (for boundary conversion).
        epsilon: float — numerical stability.

    Returns:
        advantages: (batch_size, response_length)
        returns:    (batch_size, response_length) — same as advantages for GRPO.
    """
    batch_size, response_length = eos_mask.shape
    K = substep_rewards.shape[1]

    # Group rollouts by uid (same grouping as standard GRPO)
    id2indices = defaultdict(list)
    for i in range(batch_size):
        id2indices[index[i]].append(i)

    advantages = torch.zeros(batch_size, response_length, device=eos_mask.device)

    with torch.no_grad():
        for uid, indices in id2indices.items():
            n = len(indices)
            if n <= 1:
                # Single sample: no normalization possible → zero advantage
                continue

            # (n, K) substep rewards for this group
            group_rewards = substep_rewards[indices]
            mean_k = group_rewards.mean(dim=0)  # (K,)
            std_k = group_rewards.std(dim=0)  # (K,)

            # Per-substep normalized advantages: (n, K)
            adv_k = torch.where(
                std_k.unsqueeze(0) > epsilon,
                (group_rewards - mean_k.unsqueeze(0)) / (std_k.unsqueeze(0) + epsilon),
                torch.zeros_like(group_rewards),
            )

            for j, i in enumerate(indices):
                # Build per-token substep assignment from boundary steps.
                # Tokens before first boundary → substep 0.
                # Tokens after boundary k → substep min(k+1, K-1).
                substep_ids = torch.zeros(
                    response_length, dtype=torch.long, device=eos_mask.device
                )
                for k in range(K):
                    b = int(substep_boundary_steps[i, k].item())
                    if b >= 0:
                        token_b = (b + 1) * action_token_len
                        if token_b < response_length:
                            substep_ids[token_b:] = min(k + 1, K - 1)
                substep_ids = substep_ids.clamp(max=K - 1)

                # Each token gets its substep's advantage, masked by eos
                advantages[i] = adv_k[j][substep_ids] * eos_mask[i]

    return advantages, advantages
