"""
kl_divergence_analysis.py  —  §4.2 Prediction Confidence Gap

Quantifies linguistic dependency by computing the divergence between action
distributions when language is present (P(a|V,L)) vs. absent (P(a|V,L∅)).

For L1-regression action heads the policy produces a *deterministic* 7-dim
action vector, so we approximate the distribution divergence with two metrics:

  1. Per-step L1 distance  ||a_L - a_∅||₁  (intuitive proxy)
  2. Per-dimension KL divergence over a discretised histogram built across the
     evaluation corpus (32 uniform bins in [-1, 1]).

Usage
-----
    python experiments/analysis/kl_divergence_analysis.py \
        --pretrained_checkpoint /path/to/checkpoint \
        --task_suite_name libero_goal \
        --num_samples_per_task 200 \
        --output_dir ./analysis_outputs/kl_divergence \
        --use_l1_regression True \
        --use_proprio True \
        --num_images_in_input 2

The script saves:
  - kl_results.json   : per-task and global statistics
  - kl_barplot.png    : Baseline vs APD KL divergence bar chart (if --compare_checkpoint given)
  - l1_distances.png  : Per-timestep L1 distance violin plot
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import tqdm as tqdm_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="§4.2 KL Divergence / Prediction Confidence Gap")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to the model checkpoint directory")
    parser.add_argument("--compare_checkpoint", type=str, default=None,
                        help="Optional second checkpoint for side-by-side KL comparison (e.g. baseline)")
    parser.add_argument("--task_suite_name", type=str, default="libero_goal",
                        help="LIBERO task suite name (e.g. libero_goal, libero_spatial)")
    parser.add_argument("--num_samples_per_task", type=int, default=200,
                        help="Number of (obs, instruction) pairs sampled per task")
    parser.add_argument("--output_dir", type=str, default="./analysis_outputs/kl_divergence")
    parser.add_argument("--use_l1_regression", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--use_diffusion", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_proprio", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--use_film", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--center_crop", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--unnorm_key", type=str, default="")
    parser.add_argument("--model_family", type=str, default="openvla")
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--load_in_8bit", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--load_in_4bit", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--remove_wrap", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--num_bins", type=int, default=32,
                        help="Number of histogram bins per action dimension for KL computation")
    parser.add_argument("--kl_epsilon", type=float, default=1e-8,
                        help="Smoothing epsilon for KL divergence computation")
    parser.add_argument("--checkpoint_label", type=str, default="APD",
                        help="Label for the primary checkpoint (used in plots)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Histogram-based KL divergence utilities
# ---------------------------------------------------------------------------

def build_histogram(values: np.ndarray, num_bins: int, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """Build a normalised histogram (probability distribution) from a 1-D array of values."""
    counts, _ = np.histogram(values, bins=num_bins, range=(vmin, vmax))
    total = counts.sum()
    if total == 0:
        return np.ones(num_bins, dtype=np.float64) / num_bins  # uniform fallback
    return counts.astype(np.float64) / total


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-8) -> float:
    """KL(P || Q) with Laplace smoothing to avoid log(0)."""
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Minimal config stub compatible with get_vla_action
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    pretrained_checkpoint: str
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    remove_wrap: bool = False
    task_suite_name: str = "libero_goal"
    h_decoding: bool = False
    e_decoding: bool = False
    task_label: str = ""
    save_video: bool = False


# ---------------------------------------------------------------------------
# Single-model analysis
# ---------------------------------------------------------------------------

def analyse_checkpoint(
    cfg_args,
    checkpoint_path: str,
    tasks: list,
    task_suite,
    num_bins: int,
    kl_epsilon: float,
    label: str,
) -> Dict:
    """
    For each task, collect predicted action vectors under instruction (L) and
    null instruction (L∅), then compute L1 distances and per-dim KL divergences.

    Returns a results dict:
        {
          "label": str,
          "per_task": {task_desc: {"l1_distances": [...], "kl_per_dim": [...], "mean_kl": float}},
          "global_mean_kl": float,
          "global_mean_l1": float,
        }
    """
    import torch
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from experiments.robot.openvla_utils import get_vla_action, get_processor
    from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image, quat2axisangle
    from experiments.robot.openvla_utils import get_action_head, get_proprio_projector, resize_image_for_policy
    from libero.libero import benchmark

    set_seed_everywhere(cfg_args.seed)

    cfg = AnalysisConfig(
        pretrained_checkpoint=checkpoint_path,
        model_family=cfg_args.model_family,
        use_l1_regression=cfg_args.use_l1_regression,
        use_diffusion=cfg_args.use_diffusion,
        use_film=cfg_args.use_film,
        num_images_in_input=cfg_args.num_images_in_input,
        use_proprio=cfg_args.use_proprio,
        center_crop=cfg_args.center_crop,
        lora_rank=cfg_args.lora_rank,
        unnorm_key=cfg_args.unnorm_key,
        remove_wrap=cfg_args.remove_wrap,
        task_suite_name=cfg_args.task_suite_name,
    )

    logger.info(f"[{label}] Loading model from {checkpoint_path}")
    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # Resolve unnorm_key
        if not cfg.unnorm_key:
            cfg.unnorm_key = cfg_args.task_suite_name
        if cfg.unnorm_key not in model.norm_stats:
            alt = f"{cfg.unnorm_key}_no_noops"
            if alt in model.norm_stats:
                cfg.unnorm_key = alt

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    # Collect samples across tasks
    per_task_results = {}

    task_pbar = tqdm_module.tqdm(range(len(tasks)), desc=f"[{label}] Tasks", unit="task")
    for task_id in task_pbar:
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg_args.env_img_res)

        task_pbar.set_postfix({"task": task_description[:40]})
        logger.info(f"[{label}] Task {task_id}/{len(tasks)}: {task_description[:70]}")

        logger.info(f"[{label}]   Loading initial states for task {task_id}...")
        initial_states = task_suite.get_task_init_states(task_id)
        logger.info(f"[{label}]   Loaded {len(initial_states)} initial states.")

        l1_distances: List[float] = []
        # per-dim buckets for KL
        with_lang_actions: List[np.ndarray] = []   # (N, 7)
        null_lang_actions: List[np.ndarray] = []   # (N, 7)

        samples_collected = 0
        episode_idx = 0

        sample_pbar = tqdm_module.tqdm(
            total=cfg_args.num_samples_per_task,
            desc=f"  Samples (task {task_id})",
            unit="sample",
            leave=False,
        )

        while samples_collected < cfg_args.num_samples_per_task:
            env.reset()
            init_state = initial_states[episode_idx % len(initial_states)]
            obs = env.set_init_state(init_state)

            # Collect up to num_samples_per_task frames per episode
            for step_i in range(cfg_args.num_open_loop_steps * 4):
                if samples_collected >= cfg_args.num_samples_per_task:
                    break

                # Prepare observation dict
                img = get_libero_image(obs)
                wrist_img = get_libero_wrist_image(obs)
                img_resized = resize_image_for_policy(img, resize_size)
                wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
                observation = {
                    "full_image": img_resized,
                    "wrist_image": wrist_img_resized,
                    "state": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"]
                    )),
                }

                # Action with language instruction
                try:
                    actions_with = get_vla_action(
                        cfg=cfg,
                        vla=model,
                        processor=processor,
                        obs=observation,
                        task_label=task_description,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                    )
                    a_with = actions_with[0] if isinstance(actions_with, (list, tuple)) else actions_with

                    # Action with null instruction
                    actions_null = get_vla_action(
                        cfg=cfg,
                        vla=model,
                        processor=processor,
                        obs=observation,
                        task_label="",          # L∅
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                    )
                    a_null = actions_null[0] if isinstance(actions_null, (list, tuple)) else actions_null

                    l1 = float(np.abs(a_with - a_null).sum())
                    l1_distances.append(l1)
                    with_lang_actions.append(a_with)
                    null_lang_actions.append(a_null)
                    samples_collected += 1
                    sample_pbar.update(1)
                    sample_pbar.set_postfix({"ep": episode_idx, "step": step_i, "l1": f"{l1:.3f}"})

                    # Step environment with the "with-language" action
                    from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action
                    action_exec = invert_gripper_action(normalize_gripper_action(a_with, binarize=True))
                    obs, _, done, _ = env.step(action_exec.tolist())
                    if done:
                        break

                except Exception as e:
                    logger.warning(f"  Sample error: {e}")
                    break

            episode_idx += 1
            if episode_idx >= len(initial_states) * 3:
                logger.warning(f"  Reached max episodes for task {task_id}, stopping early")
                break

        sample_pbar.close()

        if len(with_lang_actions) == 0:
            logger.warning(f"  No samples collected for task {task_id}")
            continue

        with_arr = np.stack(with_lang_actions)   # (N, 7)
        null_arr = np.stack(null_lang_actions)    # (N, 7)

        # Per-dimension KL divergence
        kl_per_dim = []
        for dim in range(with_arr.shape[1]):
            p = build_histogram(with_arr[:, dim], num_bins)
            q = build_histogram(null_arr[:, dim], num_bins)
            kl_per_dim.append(kl_divergence(p, q, kl_epsilon))

        mean_kl = float(np.mean(kl_per_dim))
        mean_l1 = float(np.mean(l1_distances))

        per_task_results[task_description] = {
            "l1_distances": l1_distances,
            "kl_per_dim": kl_per_dim,
            "mean_kl": mean_kl,
            "mean_l1": mean_l1,
            "num_samples": samples_collected,
        }
        logger.info(f"  mean_l1={mean_l1:.4f}  mean_kl={mean_kl:.4f}  (n={samples_collected})")

    global_mean_kl = float(np.mean([v["mean_kl"] for v in per_task_results.values()]))
    global_mean_l1 = float(np.mean([v["mean_l1"] for v in per_task_results.values()]))

    return {
        "label": label,
        "checkpoint": checkpoint_path,
        "per_task": per_task_results,
        "global_mean_kl": global_mean_kl,
        "global_mean_l1": global_mean_l1,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_kl_comparison(results_list: List[Dict], output_dir: str):
    """Bar chart comparing mean KL per task across checkpoints."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_tasks = sorted({t for r in results_list for t in r["per_task"]})
        x = np.arange(len(all_tasks))
        width = 0.8 / max(len(results_list), 1)

        fig, ax = plt.subplots(figsize=(max(10, len(all_tasks) * 1.2), 5))
        for i, result in enumerate(results_list):
            kl_vals = [result["per_task"].get(t, {}).get("mean_kl", 0.0) for t in all_tasks]
            ax.bar(x + i * width, kl_vals, width, label=result["label"])

        ax.set_xticks(x + width * (len(results_list) - 1) / 2)
        ax.set_xticklabels([t[:30] for t in all_tasks], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean KL Divergence KL(P(a|V,L) || P(a|V,L∅))")
        ax.set_title("§4.2 Prediction Confidence Gap — Per-Task KL Divergence")
        ax.legend()
        plt.tight_layout()

        path = os.path.join(output_dir, "kl_barplot.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved KL bar chart → {path}")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


def plot_l1_distances(results_list: List[Dict], output_dir: str):
    """Violin / box plot of per-step L1 distances."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = []
        labels = []
        for result in results_list:
            all_l1 = [v for task_data in result["per_task"].values()
                      for v in task_data["l1_distances"]]
            data_to_plot.append(all_l1)
            labels.append(result["label"])

        ax.violinplot(data_to_plot, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Per-step L1 distance  ||a_L - a_∅||₁")
        ax.set_title("§4.2 Action Shift under Language Removal")
        plt.tight_layout()

        path = os.path.join(output_dir, "l1_distances.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved L1 violin chart → {path}")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parents[2]))

    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    tasks = [task_suite.get_task(i) for i in range(task_suite.n_tasks)]

    results_list = []

    # Primary checkpoint
    result_primary = analyse_checkpoint(
        cfg_args=args,
        checkpoint_path=args.pretrained_checkpoint,
        tasks=tasks,
        task_suite=task_suite,
        num_bins=args.num_bins,
        kl_epsilon=args.kl_epsilon,
        label=args.checkpoint_label,
    )
    results_list.append(result_primary)

    # Optional comparison checkpoint (e.g. baseline)
    if args.compare_checkpoint:
        result_compare = analyse_checkpoint(
            cfg_args=args,
            checkpoint_path=args.compare_checkpoint,
            tasks=tasks,
            task_suite=task_suite,
            num_bins=args.num_bins,
            kl_epsilon=args.kl_epsilon,
            label="Baseline",
        )
        results_list.append(result_compare)

    # Save JSON results
    output_json = os.path.join(args.output_dir, "kl_results.json")
    # Convert lists to summaries for JSON serialisation
    json_out = []
    for r in results_list:
        r_copy = dict(r)
        r_copy["per_task"] = {
            task: {
                "mean_kl": v["mean_kl"],
                "mean_l1": v["mean_l1"],
                "kl_per_dim": v["kl_per_dim"],
                "num_samples": v["num_samples"],
            }
            for task, v in r["per_task"].items()
        }
        json_out.append(r_copy)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results → {output_json}")

    # Plots
    plot_kl_comparison(results_list, args.output_dir)
    plot_l1_distances(results_list, args.output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("§4.2 KL Divergence Summary")
    print("=" * 70)
    for r in results_list:
        print(f"\nModel: {r['label']}  ({r['checkpoint']})")
        print(f"  Global Mean KL  : {r['global_mean_kl']:.4f}")
        print(f"  Global Mean L1  : {r['global_mean_l1']:.4f}")
        print(f"  Per-task KL:")
        for task, v in r["per_task"].items():
            print(f"    {task[:50]:50s}  KL={v['mean_kl']:.4f}  L1={v['mean_l1']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()






