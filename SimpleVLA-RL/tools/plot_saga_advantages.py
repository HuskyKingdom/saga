#!/usr/bin/env python3
"""Visualize SAGA advantage statistics recorded during RL training.

Reads the JSONL log written by RayTrainer._write_advantage_log() and produces a
multi-panel figure showing advantage evolution, per-substep rewards, and
per-sample reward distributions over training steps.

Usage:
    python tools/plot_saga_advantages.py <path/to/saga_advantage_log.jsonl>
    python tools/plot_saga_advantages.py logs/saga_advantage_log.jsonl -o figs/saga_adv.png

The JSONL file has one record per training step with keys such as:
    step, critic_advantages_mean/max/min, saga/substep_pick_reward_mean/std,
    saga/substep_place_reward_mean/std, saga/substep_pick_adv_mean/std,
    saga/substep_place_adv_mean/std, substep_rewards (list[list[float]])
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless by default; override with -s/--show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_log(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        sys.exit(f"[plot_saga_advantages] No records found in {path}")
    records.sort(key=lambda r: r["step"])
    return records


def _get(records: list[dict], key: str, default=None) -> np.ndarray:
    """Extract a scalar field from each record, returning an ndarray."""
    vals = [r.get(key, default) for r in records]
    return np.array(vals, dtype=float)


def _fill(ax, steps, mean, std, color, label, alpha=0.18):
    ax.plot(steps, mean, color=color, marker="o", markersize=3, lw=1.4, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_saga_advantages(log_path: str, output_path: str | None = None, show: bool = False):
    records = load_log(log_path)
    steps = np.array([r["step"] for r in records])
    n = len(steps)

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # ------------------------------------------------------------------
    # Panel 1 — Overall advantage (mean ± std derived from min/max range)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    adv_mean = _get(records, "critic_advantages_mean", 0.0)
    adv_max  = _get(records, "critic_advantages_max",  0.0)
    adv_min  = _get(records, "critic_advantages_min",  0.0)
    ax1.plot(steps, adv_mean, color="steelblue", marker="o", markersize=3, lw=1.5, label="mean")
    ax1.fill_between(steps, adv_min, adv_max, color="steelblue", alpha=0.18, label="min–max")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_title("Overall Token-Level Advantage", fontsize=10)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Advantage")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2 — Per-substep binary reward mean over steps
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    pick_r_mean  = _get(records, "saga_substep_pick_reward_mean",  None)
    place_r_mean = _get(records, "saga_substep_place_reward_mean", None)
    pick_r_std   = _get(records, "saga_substep_pick_reward_std",   0.0)
    place_r_std  = _get(records, "saga_substep_place_reward_std",  0.0)
    has_saga = not np.all(np.isnan(pick_r_mean))
    if has_saga:
        _fill(ax2, steps, pick_r_mean,  pick_r_std,  "#e05252", "pick")
        _fill(ax2, steps, place_r_mean, place_r_std, "#52a852", "place")
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax2.text(0.5, 0.5, "No SAGA substep data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=9, color="gray")
    ax2.set_title("SAGA Substep Reward (pick / place)", fontsize=10)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Binary Reward — batch mean")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 3 — Per-substep normalized advantage mean ± std
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    pick_a_mean  = _get(records, "saga_substep_pick_adv_mean",  None)
    place_a_mean = _get(records, "saga_substep_place_adv_mean", None)
    pick_a_std   = _get(records, "saga_substep_pick_adv_std",   0.0)
    place_a_std  = _get(records, "saga_substep_place_adv_std",  0.0)
    if has_saga and not np.all(np.isnan(pick_a_mean)):
        _fill(ax3, steps, pick_a_mean,  pick_a_std,  "#e05252", "pick adv")
        _fill(ax3, steps, place_a_mean, place_a_std, "#52a852", "place adv")
        ax3.axhline(0, color="k", lw=0.8, ls="--")
    else:
        ax3.text(0.5, 0.5, "No SAGA adv data", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=9, color="gray")
    ax3.set_title("SAGA Per-Substep Normalized Advantage", fontsize=10)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Normalized Advantage")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 4 — Per-sample substep reward scatter (shows distribution shape)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0:2])
    has_raw = any("substep_rewards" in r for r in records)
    if has_raw:
        rng = np.random.default_rng(42)
        pick_xs, pick_ys, place_xs, place_ys = [], [], [], []
        for r in records:
            if "substep_rewards" not in r:
                continue
            sr = np.array(r["substep_rewards"])  # (B, K)
            s = r["step"]
            jitter = rng.uniform(-0.35, 0.35, size=sr.shape[0])
            pick_xs.extend((s + jitter).tolist())
            pick_ys.extend(sr[:, 0].tolist())
            if sr.shape[1] > 1:
                jitter2 = rng.uniform(-0.35, 0.35, size=sr.shape[0])
                place_xs.extend((s + jitter2).tolist())
                place_ys.extend(sr[:, 1].tolist())
        ax4.scatter(pick_xs, pick_ys,  s=12, alpha=0.55, color="#e05252", label="pick",  zorder=3)
        ax4.scatter(place_xs, place_ys, s=12, alpha=0.55, color="#52a852", label="place", zorder=3)
        ax4.set_ylim(-0.12, 1.12)
        if n <= 40:
            ax4.set_xticks(steps)
    else:
        ax4.text(0.5, 0.5, "No raw substep_rewards in log", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=9, color="gray")
    ax4.set_title("Per-Sample Substep Reward Distribution Across Steps", fontsize=10)
    ax4.set_xlabel("Training Step")
    ax4.set_ylabel("Substep Reward (binary: 0 or 1)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 5 — Returns summary
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    ret_mean = _get(records, "critic_returns_mean", 0.0)
    ret_max  = _get(records, "critic_returns_max",  0.0)
    ret_min  = _get(records, "critic_returns_min",  0.0)
    ax5.plot(steps, ret_mean, color="#7b5ea7", marker="o", markersize=3, lw=1.5, label="mean")
    ax5.fill_between(steps, ret_min, ret_max, color="#7b5ea7", alpha=0.18, label="min–max")
    ax5.axhline(0, color="k", lw=0.8, ls="--")
    ax5.set_title("Returns (mean / min–max)", fontsize=10)
    ax5.set_xlabel("Training Step")
    ax5.set_ylabel("Return")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    fig.suptitle(
        f"SAGA RL — Advantage Visualization  ({n} steps, log: {Path(log_path).name})",
        fontsize=13, fontweight="bold",
    )

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[plot_saga_advantages] Saved → {output_path}")
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot SAGA advantage statistics from a training JSONL log."
    )
    parser.add_argument("log_path", help="Path to saga_advantage_log.jsonl")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image path (PNG / PDF). Defaults to <log_path>.png",
    )
    parser.add_argument(
        "-s", "--show", action="store_true",
        help="Display the figure interactively (requires a display).",
    )
    args = parser.parse_args()

    out = args.output or str(Path(args.log_path).with_suffix(".png"))
    plot_saga_advantages(args.log_path, output_path=out, show=args.show)


if __name__ == "__main__":
    main()
