#!/bin/bash
# H100 / Enroot launcher for VANILLA GRPO RL on LIBERO.
# No SAGA, no SCOPE, no swap_objects, no distance reward — pure trajectory-level
# GRPO baseline. Counterpart of oftplus_trail.sh on AMD.
# Submits examples/run_raw_h100.sh to SLURM with pyxis.
# Run from SimpleVLA-RL root.

set -e

# =============================================================================
# Host paths — change these for your H100 environment
# =============================================================================
# Container image
export SQSH_PATH="${SQSH_PATH:-$HOME/yuhang_workspace/simplevla-rl-cuda-saga.sqsh}"

# SFT starting checkpoint (mounted RW; overwrite_vla_ckpt_utils.sh patches in-place)
export HOST_SFT_MODEL_DIR="${HOST_SFT_MODEL_DIR:-$HOME/yuhang_workspace/landmarked_ckpoints/oft_plus_discrete}"

# Where this run's checkpoints will land on the host
export HOST_CKPT_DIR="${HOST_CKPT_DIR:-$HOME/yuhang_workspace/landmarked_ckpoints/raw_ckpts}"

# Optional: bind host repo over the baked-in /workspace copy for dev iteration
export HOST_REPO_OVERRIDE="${HOST_REPO_OVERRIDE:-$HOME/yuhang_workspace/openvla-oft-yhs}"

# WandB — hard-coded (see saga_rl_trail_h100.sh for rationale)
export WANDB_API_KEY="0bdbd99b1136358467ed2d03e9a6ba5a5b2a11a8"

# =============================================================================
# Run config (matches oftplus_trail.sh — vanilla GRPO, no SAGA/SCOPE)
# =============================================================================
export DATASET_NAME="libero_4_task_suites"
export EXPERIMENT_NAME="raw-rl-libero-h100"
export PROJECT_NAME="openvla-oft-rl"
export NUM_GPUS=8

# Autoregressive vs parallel decoding — keep matching your SFT model:
#   oft_plus_discrete keeps PARALLEL decoding but uses DISCRETE tokens (no L1
#   regression head, but still emits 8-step action chunk in one forward).
#   So USE_AUTOREGRESSIVE="False" is correct for oft_plus_discrete.
export USE_AUTOREGRESSIVE="False"

# GRPO baseline does NOT use SAGA / SWAP / DIST reward — those are different
# experimental knobs. If you want to compare a curriculum-swap baseline, use
# the saga script instead and set ADV_ESTIMATOR=grpo + SWAP_OBJECTS=True.

# Continue training from a non-zero step (e.g. resume); 0 = train from scratch
export TRAINER_INITIAL_GLOBAL_STEPS="${TRAINER_INITIAL_GLOBAL_STEPS:-0}"

# Batch invariants:
#   ACTOR_PPO_MINI_BATCH_SIZE  <= DATA_TRAIN_BATCH_SIZE * DATA_N_SAMPLES
#   ACTOR_TRAJ_MINI_BATCH_SIZE <= DATA_TRAIN_BATCH_SIZE
# Defaults below match oftplus_trail.sh (DATA_N_SAMPLES=4 from run_raw_h100.sh).
# H100 has more headroom than MI300X — feel free to bump for faster convergence.
export DATA_TRAIN_BATCH_SIZE="64"
export ACTOR_PPO_MINI_BATCH_SIZE="128"   # <= 64 * 4 = 256
export ACTOR_TRAJ_MINI_BATCH_SIZE="16"   # <= 64

# Optional: total training epochs (default 100 in run_raw_h100.sh)
# export TRAINER_TOTAL_EPOCHS="75"

# =============================================================================
# SLURM submit
# =============================================================================
sbatch examples/run_raw_h100.sh
