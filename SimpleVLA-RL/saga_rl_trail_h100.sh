#!/bin/bash
# H100 / Enroot equivalent of saga_rl_trail.sh.
# Submits examples/run_saga_h100.sh to SLURM with pyxis.
# Run from SimpleVLA-RL root.

set -e

# =============================================================================
# Host paths — change these for your H100 environment
# =============================================================================
# Path to the .sqsh produced by SimpleVLA-RL/docker/build_h100.sh
export SQSH_PATH="${SQSH_PATH:-$HOME/yuhang_workspace/simplevla-rl-cuda-saga.sqsh}"

# Host directory containing the SFT checkpoint (e.g. oft_plus_discrete/)
# Mounted read-only into the container.
export HOST_SFT_MODEL_DIR="$HOME/yuhang_workspace/landmarked_ckpoints/oft_plus_discrete"

# Where checkpoints from this run will land on the host.
export HOST_CKPT_DIR="${HOST_CKPT_DIR:-$HOME/yuhang_workspace/landmarked_ckpoints/saga_ckpts_woswap}"

# Host path to APD_plans_scaled.json. Mounted read-only into the container.
export HOST_APD_PLANS_FILE="${HOST_APD_PLANS_FILE:-${HOME}/yuhang_workspace/openvla-oft-yhs/APD_plans_scaled.json}"

# Optional: bind-mount host repo over the baked-in /workspace copy for dev.
# Leave empty to use the code baked into the image.
export HOST_REPO_OVERRIDE="$HOME/yuhang_workspace/openvla-oft-yhs"

# WandB — hard-coded so we don't depend on align.json being readable inside
# the container, and so SBATCH env-inheritance quirks don't drop the key.
# WANDB_DIR is set to a /tmp path INSIDE the container (in run_saga_h100.sh's
# INNER_CMD), not here — host paths wouldn't exist inside the rootfs anyway.
export WANDB_API_KEY="0bdbd99b1136358467ed2d03e9a6ba5a5b2a11a8"

# =============================================================================
# Run config (matches saga_rl_trail.sh)
# =============================================================================
export DATASET_NAME="libero_4_task_suites"
export EXPERIMENT_NAME="saga-rl-libero-h100"
export PROJECT_NAME="openvla-oft-rl"
export NUM_GPUS=8

export USE_AUTOREGRESSIVE="False"

export SWAP_OBJECTS="False"
export SWAP_DISTANCE_START="0.08"
export SWAP_DISTANCE_END="0.40"
export SWAP_CURRICULUM_STEPS="12000"

export DIST_REWARD_COEF="0.0"
export DIST_REWARD_SIGMA="0.05"

export ADV_ESTIMATOR="saga"

export TRAINER_INITIAL_GLOBAL_STEPS="${TRAINER_INITIAL_GLOBAL_STEPS:-0}"

export DATA_TRAIN_BATCH_SIZE="64"
export ACTOR_PPO_MINI_BATCH_SIZE="128"   # <= DATA_TRAIN_BATCH_SIZE * n_samples (8*4=32)
export ACTOR_TRAJ_MINI_BATCH_SIZE="16"   # <= DATA_TRAIN_BATCH_SIZE

# =============================================================================
# SLURM submit
# =============================================================================
# If your cluster needs --partition / --account, set them via SBATCH_PARTITION
# / SBATCH_ACCOUNT env vars or pass --partition=... here.
sbatch examples/run_saga_h100.sh
