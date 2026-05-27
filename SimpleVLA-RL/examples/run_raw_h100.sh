#!/bin/bash
#SBATCH --job-name=raw-rl-libero-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=p03
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/raw_h100_%j.out
#SBATCH --error=slurm/logs/raw_h100_%j.err
# Vanilla GRPO RL on NVIDIA H100 (single node, 8 GPUs) — pyxis + enroot.
# No SAGA, no SCOPE, no swap_objects, no distance reward — pure trajectory-level
# GRPO with success-only reward. Counterpart of run_openvla_oft_plus_rl_libero.sh.
# Submit from SimpleVLA-RL root:  sbatch examples/run_raw_h100.sh

set -ex

# =============================================================================
# 1. CONTAINER (pyxis fields — set these for your cluster)
# =============================================================================
SQSH_PATH="${SQSH_PATH:-${HOME}/yuhang_workspace/simplevla-rl-cuda-saga.sqsh}"

HOST_SFT_MODEL_DIR="${HOST_SFT_MODEL_DIR:-${HOME}/yuhang_workspace/landmarked_ckpoints/oft_plus_discrete}"
HOST_CKPT_DIR="${HOST_CKPT_DIR:-${HOME}/yuhang_workspace/landmarked_ckpoints/raw_ckpts}"
HOST_REPO_OVERRIDE="${HOST_REPO_OVERRIDE:-}"

mkdir -p "$HOST_CKPT_DIR"

if [ ! -f "$SQSH_PATH" ]; then
    echo "Error: container image not found: $SQSH_PATH"; exit 1
fi
if [ ! -d "$HOST_SFT_MODEL_DIR" ]; then
    echo "Error: HOST_SFT_MODEL_DIR not found: $HOST_SFT_MODEL_DIR"; exit 1
fi

# Container-internal paths
REPO_ROOT_C="/workspace/openvla-oft-yhs"
SFT_MODEL_PATH_C="/mnt/sft_models/$(basename "$HOST_SFT_MODEL_DIR")"
CKPT_PATH_C="/mnt/ckpt"
ALIGN_PATH_C="${REPO_ROOT_C}/SimpleVLA-RL/align.json"

# pyxis bind list — note SFT is RW because overwrite_vla_ckpt_utils.sh patches
# in-place. No APD plans mount (vanilla GRPO doesn't need it).
CONTAINER_MOUNTS="${HOST_SFT_MODEL_DIR}:${SFT_MODEL_PATH_C}"
CONTAINER_MOUNTS+=",${HOST_CKPT_DIR}:${CKPT_PATH_C}"
if [ -n "$HOST_REPO_OVERRIDE" ]; then
    CONTAINER_MOUNTS+=",${HOST_REPO_OVERRIDE}:${REPO_ROOT_C}"
fi

# =============================================================================
# 2. TRAINING RUN CONFIG
# =============================================================================
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-raw-rl-libero}"
DATASET_NAME="${DATASET_NAME:-libero_4_task_suites}"
VLA_NAME="${VLA_NAME:-openvla-oft}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"

SFT_MODEL_PATH="${SFT_MODEL_PATH_C}"
CKPT_PATH="${CKPT_PATH_C}"
ALIGN_PATH="${ALIGN_PATH_C}"

# =============================================================================
# 3. DATA / ACTOR / ROLLOUT CONFIG (Hydra overrides)
# =============================================================================
DATA_NUM_TRIALS_PER_TASK="${DATA_NUM_TRIALS_PER_TASK:-50}"
DATA_N_SAMPLES="${DATA_N_SAMPLES:-4}"
DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-64}"
DATA_VAL_BATCH_SIZE="${DATA_VAL_BATCH_SIZE:-496}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-256}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-128}"

ACTOR_LR="${ACTOR_LR:-5e-6}"
ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-128}"
ACTOR_PPO_MICRO_BATCH_SIZE="${ACTOR_PPO_MICRO_BATCH_SIZE:-$NUM_GPUS}"
ACTOR_TRAJ_MINI_BATCH_SIZE="${ACTOR_TRAJ_MINI_BATCH_SIZE:-16}"
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-1}"
ROLLOUT_VAL_MICRO_BATCH_SIZE="${ROLLOUT_VAL_MICRO_BATCH_SIZE:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.6}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-32}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.9}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-32}"

# =============================================================================
# 4. REWARD CONFIG (Hydra verifier.*)
# =============================================================================
USE_AUTOREGRESSIVE="${USE_AUTOREGRESSIVE:-False}"
VERIFIER_REWARD_COEF="${VERIFIER_REWARD_COEF:-5}"
KL_COEF="${KL_COEF:-0.00}"

# =============================================================================
# 5. TRAINER SCHEDULE
# =============================================================================
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-25}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-4}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-100}"
TRAINER_INITIAL_GLOBAL_STEPS="${TRAINER_INITIAL_GLOBAL_STEPS:-0}"

# =============================================================================
# 6. RUNTIME ENV (CUDA equivalents of run_openvla_oft_plus_rl_libero.sh)
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
# GPU EGL rendering on H100 (NVIDIA driver hooks). graphics cap is required.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
export RAY_object_store_memory="${RAY_object_store_memory:-20000000000}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"

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

echo "Vanilla GRPO RL (H100/Enroot) — image: $SQSH_PATH"
echo "  SFT (in-container): $SFT_MODEL_PATH"
echo "  Mounts: $CONTAINER_MOUNTS"
echo "  R_task coef=$VERIFIER_REWARD_COEF  KL=$KL_COEF  adv=grpo"

# =============================================================================
# 8. INNER COMMAND — runs inside the enroot container
#     No saga.*, no swap_*, no dist_reward_* flags. adv_estimator=grpo.
# =============================================================================
INNER_CMD=$(cat <<EOF
set -ex
# HOME=/root because pyxis leaks the host HOME into the container (RO).
# /root/.libero/config.yaml was pre-created in the Dockerfile.
export HOME=/root
# wandb writes locally before syncing. /tmp is writable on enroot tmpfs.
# Path hard-coded — heredoc expands $WANDB_DIR on the host (where it's empty)
# before INNER_CMD runs in the container.
export WANDB_DIR=/tmp/wandb_runs
cd "$REPO_ROOT_C/SimpleVLA-RL"
mkdir -p slurm/logs "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME" /tmp/wandb_runs

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
    actor_rollout_ref.actor.use_autoregressive=$USE_AUTOREGRESSIVE \\
    actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_LOG_PROB_MICRO_BATCH_SIZE \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    algorithm.adv_estimator=grpo \\
    algorithm.kl_ctrl.kl_coef=$KL_COEF \\
    algorithm.adv_params.verifier_gamma=1.0 \\
    algorithm.adv_params.reward_model_gamma=1.0 \\
    verifier.reward_coef=$VERIFIER_REWARD_COEF \\
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
