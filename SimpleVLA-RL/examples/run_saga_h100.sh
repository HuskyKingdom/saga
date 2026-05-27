#!/bin/bash
#SBATCH --job-name=saga-rl-libero-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=p03
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/saga_h100_%j.out
#SBATCH --error=slurm/logs/saga_h100_%j.err
# SAGA RL on NVIDIA H100 (single node, 8 GPUs) — pyxis + enroot.
# Submit from SimpleVLA-RL root:  sbatch examples/run_saga_h100.sh
# Edit SBATCH --partition / --account / sqsh path below for your cluster.

set -ex

# =============================================================================
# 1. CONTAINER (pyxis fields — set these for your cluster)
# =============================================================================
# Path to the squashfs image built by docker/build_h100.sh and uploaded.
SQSH_PATH="${SQSH_PATH:-${HOME}/containers/simplevla-rl-cuda-saga.sqsh}"

# Where the host puts everything that the container should see at runtime.
# Defaults assume the .sqsh is self-contained (code is baked in at /workspace),
# and only the SFT checkpoint, APD plans file, and CKPT save dir are mounted.
HOST_SFT_MODEL_DIR="${HOST_SFT_MODEL_DIR:-${HOME}/sft_models}"
HOST_CKPT_DIR="${HOST_CKPT_DIR:-${HOME}/saga_ckpts}"
HOST_APD_PLANS_FILE="${HOST_APD_PLANS_FILE:-${HOME}/APD_plans_scaled.json}"
# Optional: bind-mount your host repo over the baked-in /workspace copy for dev
HOST_REPO_OVERRIDE="${HOST_REPO_OVERRIDE:-}"   # e.g. /home/yuhang/openvla-oft-yhs

mkdir -p "$HOST_CKPT_DIR"

if [ ! -f "$SQSH_PATH" ]; then
    echo "Error: container image not found: $SQSH_PATH"
    echo "Build it with: bash SimpleVLA-RL/docker/build_h100.sh   then scp the .sqsh"
    exit 1
fi
if [ ! -d "$HOST_SFT_MODEL_DIR" ]; then
    echo "Error: HOST_SFT_MODEL_DIR not found: $HOST_SFT_MODEL_DIR"; exit 1
fi
if [ ! -f "$HOST_APD_PLANS_FILE" ]; then
    echo "Error: HOST_APD_PLANS_FILE not found: $HOST_APD_PLANS_FILE"; exit 1
fi

# Container-internal paths (stable, used in the python command below)
REPO_ROOT_C="/workspace/openvla-oft-yhs"
SFT_MODEL_PATH_C="/mnt/sft_models/$(basename "$HOST_SFT_MODEL_DIR")"   # whole dir bound
CKPT_PATH_C="/mnt/ckpt"
APD_PLANS_PATH_C="/mnt/apd_plans.json"
ALIGN_PATH_C="${REPO_ROOT_C}/SimpleVLA-RL/align.json"

# Build pyxis bind list. Each "host:container[:ro]" pair separated by commas.
# SFT is mounted RW because overwrite_vla_ckpt_utils.sh patches a handful of
# prismatic files in-place (mirrors AMD's apptainer --bind default RW). The
# host SFT dir will be modified after the first run; if you want to preserve
# the pristine checkpoint, point HOST_SFT_MODEL_DIR at a one-time host copy.
CONTAINER_MOUNTS="${HOST_SFT_MODEL_DIR}:${SFT_MODEL_PATH_C}"
CONTAINER_MOUNTS+=",${HOST_CKPT_DIR}:${CKPT_PATH_C}"
CONTAINER_MOUNTS+=",${HOST_APD_PLANS_FILE}:${APD_PLANS_PATH_C}:ro"
if [ -n "$HOST_REPO_OVERRIDE" ]; then
    CONTAINER_MOUNTS+=",${HOST_REPO_OVERRIDE}:${REPO_ROOT_C}"
fi

# =============================================================================
# 2. TRAINING RUN CONFIG
# =============================================================================
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-saga-rl-libero}"
DATASET_NAME="${DATASET_NAME:-libero_4_task_suites}"
VLA_NAME="${VLA_NAME:-openvla-oft}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"

# Inside-container values — these go into Hydra overrides, not host paths
SFT_MODEL_PATH="${SFT_MODEL_PATH_C}"
CKPT_PATH="${CKPT_PATH_C}"
APD_PLANS_PATH="${APD_PLANS_PATH_C}"
ALIGN_PATH="${ALIGN_PATH_C}"

# =============================================================================
# 3. DATA / ACTOR / ROLLOUT (Hydra overrides — same defaults as run_saga.sh)
# =============================================================================
DATA_NUM_TRIALS_PER_TASK="${DATA_NUM_TRIALS_PER_TASK:-50}"
DATA_N_SAMPLES="${DATA_N_SAMPLES:-4}"
DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-8}"
DATA_VAL_BATCH_SIZE="${DATA_VAL_BATCH_SIZE:-496}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-256}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-128}"

ACTOR_LR="${ACTOR_LR:-5e-6}"
ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-32}"
ACTOR_PPO_MICRO_BATCH_SIZE="${ACTOR_PPO_MICRO_BATCH_SIZE:-$NUM_GPUS}"
ACTOR_TRAJ_MINI_BATCH_SIZE="${ACTOR_TRAJ_MINI_BATCH_SIZE:-8}"
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-1}"
ROLLOUT_VAL_MICRO_BATCH_SIZE="${ROLLOUT_VAL_MICRO_BATCH_SIZE:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.6}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-32}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-32}"

# =============================================================================
# 4. REWARD / SAGA CONFIG
# =============================================================================
USE_AUTOREGRESSIVE="${USE_AUTOREGRESSIVE:-False}"
SWAP_OBJECTS="${SWAP_OBJECTS:-True}"
SWAP_DISTANCE_START="${SWAP_DISTANCE_START:-0.08}"
SWAP_DISTANCE_END="${SWAP_DISTANCE_END:-0.40}"
SWAP_CURRICULUM_STEPS="${SWAP_CURRICULUM_STEPS:-12000}"
VERIFIER_REWARD_COEF="${VERIFIER_REWARD_COEF:-5}"
KL_COEF="${KL_COEF:-0.00}"
DIST_REWARD_COEF="${DIST_REWARD_COEF:-0.0}"
DIST_REWARD_SIGMA="${DIST_REWARD_SIGMA:-0.05}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-saga}"

# =============================================================================
# 5. TRAINER SCHEDULE
# =============================================================================
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-25}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-4}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-100}"
TRAINER_INITIAL_GLOBAL_STEPS="${TRAINER_INITIAL_GLOBAL_STEPS:-0}"

# =============================================================================
# 6. RUNTIME ENV (passed into the container)
# =============================================================================
# CUDA equivalents of the ROCm vars in run_saga.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
# GPU-accelerated headless rendering via NVIDIA EGL (replaces AMD's CPU-side
# osmesa fallback). NVIDIA_DRIVER_CAPABILITIES MUST include "graphics" so the
# enroot/pyxis NVIDIA hook injects libEGL_nvidia.so.0; without it MuJoCo falls
# back to CPU silently and rollouts run an order of magnitude slower.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
export RAY_object_store_memory="${RAY_object_store_memory:-20000000000}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"

# WandB key — read from align.json if not set, mirroring run_saga.sh
if [ -z "${WANDB_API_KEY:-}" ] || [ "$WANDB_API_KEY" = "YOUR WANDB KEY" ]; then
    if [ -n "$HOST_REPO_OVERRIDE" ] && [ -f "$HOST_REPO_OVERRIDE/SimpleVLA-RL/align.json" ]; then
        WANDB_API_KEY=$(python3 -c "import json; d=json.load(open('$HOST_REPO_OVERRIDE/SimpleVLA-RL/align.json')); print(d.get('env_vars',{}).get('WANDB_API_KEY',''))")
    fi
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# =============================================================================
# 7. DERIVED
# =============================================================================
EXPERIMENT_NAME_ESC=$(printf '%s' "$EXPERIMENT_NAME" | sed 's/"/\\"/g')
DEFAULT_LOCAL_DIR="${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

echo "SAGA RL (H100/Enroot) — image: $SQSH_PATH"
echo "  SFT (in-container): $SFT_MODEL_PATH"
echo "  Mounts: $CONTAINER_MOUNTS"

# =============================================================================
# 8. INNER COMMAND — runs inside the enroot container
#    Keeps the same Hydra overrides as run_saga.sh; container_path versions of
#    SFT/CKPT/APD/ALIGN replace the host paths.
# =============================================================================
INNER_CMD=$(cat <<EOF
set -ex
# Force HOME=/root inside the container. SLURM --export=ALL leaks the host
# user's HOME into the container, but on this cluster that path is read-only
# (or non-existent) inside the rootfs — LIBERO's __init__.py runs
# os.makedirs(expanduser("~") + "/.libero") at import time and crashes.
# /root/.libero/config.yaml was pre-created in the Dockerfile, so HOME=/root
# makes LIBERO see existing config and skip the mkdir.
export HOME=/root
# wandb writes its run dir locally before syncing to the cloud. /tmp is
# writable on the enroot tmpfs; default ./wandb (cwd) is the read-only image
# rootfs. Path hard-coded — \$WANDB_DIR isn't expanded in this heredoc context
# (host shell evaluates it before INNER_CMD runs in the container).
export WANDB_DIR=/tmp/wandb_runs
cd "$REPO_ROOT_C/SimpleVLA-RL"
mkdir -p slurm/logs "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME" /tmp/wandb_runs

# Patch the SFT checkpoint's vla_modeling_utils.py (same as ROCm path)
bash examples/overwrite_vla_ckpt_utils.sh "$SFT_MODEL_PATH"

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \\
    data.task_suite_name='$DATASET_NAME' \\
    data.num_trials_per_task=$DATA_NUM_TRIALS_PER_TASK \\
    data.n_samples=$DATA_N_SAMPLES \\
    data.filter_accuracy=True \\
    data.accuracy_lower_bound=0.1 \\
    data.accuracy_upper_bound=0.9 \\
    data.oversample_factor=1 \\
    data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \\
    data.val_batch_size=$DATA_VAL_BATCH_SIZE \\
    data.max_prompt_length=$DATA_MAX_PROMPT_LENGTH \\
    data.max_response_length=$DATA_MAX_RESPONSE_LENGTH \\
    actor_rollout_ref.model.path="$SFT_MODEL_PATH" \\
    actor_rollout_ref.model.vla='$VLA_NAME' \\
    actor_rollout_ref.model.action_token_len=7 \\
    actor_rollout_ref.model.action_chunks_len=8 \\
    actor_rollout_ref.model.proprio_dim=8 \\
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \\
    actor_rollout_ref.actor.optim.warmup_style=constant \\
    actor_rollout_ref.actor.ppo_mini_batch_size=$ACTOR_PPO_MINI_BATCH_SIZE \\
    actor_rollout_ref.actor.ppo_micro_batch_size=$ACTOR_PPO_MICRO_BATCH_SIZE \\
    actor_rollout_ref.actor.use_dynamic_bsz=False \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.actor.grad_clip=1 \\
    actor_rollout_ref.actor.clip_ratio_high=0.28 \\
    actor_rollout_ref.actor.clip_ratio_low=0.2 \\
    actor_rollout_ref.actor.num_images_in_input=1 \\
    actor_rollout_ref.actor.traj_mini_batch_size=$ACTOR_TRAJ_MINI_BATCH_SIZE \\
    actor_rollout_ref.model.enable_gradient_checkpointing=False \\
    actor_rollout_ref.model.use_remove_padding=False \\
    actor_rollout_ref.actor.entropy_coeff=0. \\
    actor_rollout_ref.rollout.num_images_in_input=1 \\
    actor_rollout_ref.rollout.use_proprio=False \\
    actor_rollout_ref.rollout.val_micro_batch_size=$ROLLOUT_VAL_MICRO_BATCH_SIZE \\
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \\
    actor_rollout_ref.rollout.experiment_name="$EXPERIMENT_NAME_ESC" \\
    actor_rollout_ref.rollout.micro_batch_size=$ROLLOUT_MICRO_BATCH_SIZE \\
    actor_rollout_ref.rollout.unnorm_key='$DATASET_NAME' \\
    actor_rollout_ref.rollout.model_family=openvla \\
    actor_rollout_ref.rollout.task_suite_name='$DATASET_NAME' \\
    actor_rollout_ref.rollout.num_steps_wait=10 \\
    actor_rollout_ref.rollout.pretrained_checkpoint="$SFT_MODEL_PATH" \\
    actor_rollout_ref.rollout.center_crop=True \\
    actor_rollout_ref.rollout.max_prompt_length=512 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=hf \\
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \\
    actor_rollout_ref.rollout.use_autoregressive=$USE_AUTOREGRESSIVE \\
    actor_rollout_ref.rollout.swap_objects=$SWAP_OBJECTS \\
    actor_rollout_ref.rollout.swap_min_distance=$SWAP_DISTANCE_START \\
    actor_rollout_ref.rollout.swap_max_distance=$SWAP_DISTANCE_END \\
    actor_rollout_ref.rollout.swap_curriculum_steps=$SWAP_CURRICULUM_STEPS \\
    actor_rollout_ref.actor.use_autoregressive=$USE_AUTOREGRESSIVE \\
    actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_LOG_PROB_MICRO_BATCH_SIZE \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    saga.enabled=True \\
    saga.apd_plans_path="$APD_PLANS_PATH" \\
    algorithm.kl_ctrl.kl_coef=$KL_COEF \\
    algorithm.adv_estimator=$ADV_ESTIMATOR \\
    algorithm.adv_params.verifier_gamma=1.0 \\
    algorithm.adv_params.reward_model_gamma=1.0 \\
    verifier.reward_coef=$VERIFIER_REWARD_COEF \\
    verifier.dist_reward_coef=$DIST_REWARD_COEF \\
    verifier.dist_reward_sigma=$DIST_REWARD_SIGMA \\
    trainer.logger="[console,wandb]" \\
    trainer.project_name='$PROJECT_NAME' \\
    trainer.experiment_name="$EXPERIMENT_NAME_ESC" \\
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \\
    trainer.n_gpus_per_node=$NUM_GPUS \\
    trainer.nnodes=$NUM_NODES \\
    trainer.save_freq=$TRAINER_SAVE_FREQ \\
    trainer.test_freq=$TRAINER_TEST_FREQ \\
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \\
    trainer.initial_global_steps=$TRAINER_INITIAL_GLOBAL_STEPS \\
    trainer.val_only=False \\
    trainer.runtime_env="$ALIGN_PATH" \\
    trainer.wandb_mode=online \\
    trainer.val_before_train=False
EOF
)

# =============================================================================
# 9. LAUNCH via srun + pyxis
# =============================================================================
srun \
    --container-image="$SQSH_PATH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="${REPO_ROOT_C}/SimpleVLA-RL" \
    --no-container-mount-home \
    --export=ALL \
    bash -c "$INNER_CMD"
