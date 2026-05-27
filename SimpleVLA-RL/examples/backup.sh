#!/bin/bash
#SBATCH --job-name=apd-rl-libero
#SBATCH --partition=mi3008xl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=184
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/oft_rl_libero_%j.out
#SBATCH --error=slurm/logs/oft_rl_libero_%j.err
# OpenVLA-OFT GRPO on LIBERO — runs inside simplevla-rl-rocm.sif (Apptainer).
# Submit from SimpleVLA-RL root:  sbatch examples/run_openvla_oft_rl_libero_apptainer.sh
# Example: SFT_MODEL_PATH=checkpoints/openvla-oft-sft-full CKPT_PATH=checkpoints sbatch examples/run_openvla_oft_rl_libero_apptainer.sh

set -ex

# =============================================================================
# 1. PATHS & CONTAINER
# =============================================================================
# Repo root: from Slurm submit dir when using sbatch, else from script path
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$REPO_ROOT/simplevla-rl-rocm.sif}"

if [ ! -d "$REPO_ROOT/examples" ] || [ ! -f "$REPO_ROOT/verl/trainer/main_ppo.py" ]; then
    echo "Error: repo root not found: $REPO_ROOT. Run sbatch from SimpleVLA-RL root."
    exit 1
fi
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image not found: $APPTAINER_IMAGE"
    echo "Set APPTAINER_IMAGE or build: apptainer build simplevla-rl-rocm.sif simplevla-rl-rocm.def"
    exit 1
fi

# =============================================================================
# 2. TRAINING RUN CONFIG (edit these or pass via env when sbatch)
# =============================================================================
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
# Experiment name: no spaces, or Hydra will parse as multiple overrides
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-libero}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-YOUR SFT_MODEL_PATH}"
CKPT_PATH="${CKPT_PATH:-THE PATH YOU WANT TO SAVE YOUR CKPT}"
# Dataset: libero_10 (libero_Long), libero_90, libero_spatial, libero_object, libero_goal
DATASET_NAME="${DATASET_NAME:-libero_10}"
VLA_NAME="${VLA_NAME:-openvla-oft}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
ALIGN_PATH="${ALIGN_PATH:-$REPO_ROOT/align.json}"

# =============================================================================
# 3. DATA CONFIG (Hydra data.*)
# =============================================================================
DATA_NUM_TRIALS_PER_TASK="${DATA_NUM_TRIALS_PER_TASK:-50}"
DATA_N_SAMPLES="${DATA_N_SAMPLES:-8}"
DATA_TRAIN_BATCH_SIZE="${DATA_TRAIN_BATCH_SIZE:-64}"
DATA_VAL_BATCH_SIZE="${DATA_VAL_BATCH_SIZE:-496}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-256}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-128}"

# =============================================================================
# 4. ACTOR / ROLLOUT / REF CONFIG (Hydra actor_rollout_ref.*)
# =============================================================================
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
# 5. TRAINER SCHEDULE (Hydra trainer.*)
# =============================================================================
TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-25}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-4}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-100}"

# =============================================================================
# 6. RUNTIME ENVIRONMENT (NCCL, Ray, ROCm, caches, WandB)
#     These are passed into the container via apptainer --env in §10 (single source).
# =============================================================================
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export RAY_object_store_memory="${RAY_object_store_memory:-20000000000}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$REPO_ROOT/.cache/huggingface}"
mkdir -p "$HF_CACHE_DIR"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
mkdir -p "$TRITON_CACHE_DIR"

# WandB: prefer env, else read from align.json
if [ -z "${WANDB_API_KEY:-}" ] || [ "$WANDB_API_KEY" = "YOUR WANDB KEY" ]; then
    if [ -f "$REPO_ROOT/align.json" ]; then
        WANDB_API_KEY=$(python3 -c "import json; d=json.load(open('$REPO_ROOT/align.json')); print(d.get('env_vars',{}).get('WANDB_API_KEY',''))")
    fi
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# LD_LIBRARY_PATH: must match container %environment (simplevla-rl-rocm.def). Include libibverbs so driver libbnxt_re-rdmav34.so (if you copied it in in .def) is found at runtime.
export LD_LIBRARY_PATH="/usr/local/lib:/opt/rocm/lib:/opt/ompi/lib:/opt/ucx/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs"

ROCM_ROOT="/opt/rocm"
HIP_VERSION_CONTAINER="${HIP_VERSION:-7.1.0}"
HIP_PATH_CONTAINER="${ROCM_ROOT}"

# =============================================================================
# 7. VALIDATION (must set real paths before running)
# =============================================================================
if [[ "$SFT_MODEL_PATH" == *"YOUR"* ]] || [[ ! -d "$SFT_MODEL_PATH" ]]; then
    echo "Error: Set a real SFT_MODEL_PATH (script or env). Example: SFT_MODEL_PATH=checkpoints/openvla-oft-sft-full"
    exit 1
fi
if [[ "$CKPT_PATH" == *"THE PATH"* ]]; then
    echo "Error: Set a real CKPT_PATH (e.g. checkpoints)."
    exit 1
fi

# =============================================================================
# 8. DERIVED (used in INNER_CMD / Hydra overrides)
# =============================================================================
# Escape EXPERIMENT_NAME for double-quoted Hydra overrides (spaces would break parsing)
EXPERIMENT_NAME_ESC=$(printf '%s' "$EXPERIMENT_NAME" | sed 's/"/\\"/g')
# Checkpoint save dir: CKPT_PATH/PROJECT_NAME/EXPERIMENT_NAME
DEFAULT_LOCAL_DIR="${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

echo "GRPO (Apptainer) — SFT: $SFT_MODEL_PATH  Experiment: $EXPERIMENT_NAME"

# =============================================================================
# 9. INNER_CMD: command run inside the container (env from §10 --env only)
# =============================================================================
INNER_CMD="set -ex; \
cd \"$REPO_ROOT\"; \
mkdir -p my_rl_pkgs; \
export PYTHONPATH=\"$REPO_ROOT/my_rl_pkgs:\$PYTHONPATH\"; \
pip install --target=\"$REPO_ROOT/my_rl_pkgs\" ray codetiming \"protobuf<5\" -U wandb; \
bash examples/overwrite_vla_ckpt_utils.sh \"$SFT_MODEL_PATH\"; \
HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  data.task_suite_name='$DATASET_NAME' \
  data.num_trials_per_task=$DATA_NUM_TRIALS_PER_TASK \
  data.n_samples=$DATA_N_SAMPLES \
  data.filter_accuracy=True \
  data.accuracy_lower_bound=0.1 \
  data.accuracy_upper_bound=0.9 \
  data.oversample_factor=1 \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.val_batch_size=$DATA_VAL_BATCH_SIZE \
  data.max_prompt_length=$DATA_MAX_PROMPT_LENGTH \
  data.max_response_length=$DATA_MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path=\"$SFT_MODEL_PATH\" \
  actor_rollout_ref.model.vla='$VLA_NAME' \
  actor_rollout_ref.model.action_token_len=7 \
  actor_rollout_ref.model.action_chunks_len=8 \
  actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ACTOR_PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size=$ACTOR_PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.grad_clip=1 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.num_images_in_input=1 \
  actor_rollout_ref.actor.traj_mini_batch_size=$ACTOR_TRAJ_MINI_BATCH_SIZE \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.entropy_coeff=0. \
  actor_rollout_ref.rollout.num_images_in_input=1 \
  actor_rollout_ref.rollout.use_proprio=False \
  actor_rollout_ref.rollout.val_micro_batch_size=$ROLLOUT_VAL_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
  actor_rollout_ref.rollout.experiment_name=\"$EXPERIMENT_NAME_ESC\" \
  actor_rollout_ref.rollout.micro_batch_size=$ROLLOUT_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.unnorm_key='$DATASET_NAME' \
  actor_rollout_ref.rollout.model_family=openvla \
  actor_rollout_ref.rollout.task_suite_name='$DATASET_NAME' \
  actor_rollout_ref.rollout.num_steps_wait=10 \
  actor_rollout_ref.rollout.pretrained_checkpoint=\"$SFT_MODEL_PATH\" \
  actor_rollout_ref.rollout.center_crop=True \
  actor_rollout_ref.rollout.max_prompt_length=512 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  trainer.logger=\"[console,wandb]\" \
  trainer.project_name='$PROJECT_NAME' \
  trainer.experiment_name=\"$EXPERIMENT_NAME_ESC\" \
  trainer.default_local_dir=\"$DEFAULT_LOCAL_DIR\" \
  trainer.n_gpus_per_node=$NUM_GPUS \
  trainer.nnodes=$NUM_NODES \
  trainer.save_freq=$TRAINER_SAVE_FREQ \
  trainer.test_freq=$TRAINER_TEST_FREQ \
  trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
  trainer.val_only=False \
  algorithm.adv_estimator=grpo \
  algorithm.adv_params.verifier_gamma=1.0 \
  algorithm.adv_params.reward_model_gamma=1.0 \
  trainer.runtime_env=\"$ALIGN_PATH\" \
  trainer.wandb_mode=online \
  trainer.val_before_train=True"

# =============================================================================
# 10. RUN CONTAINER (env from §6 — single source, no duplicate in INNER_CMD)
#     --cleanenv: avoid inheriting host env (e.g. LD_LIBRARY_PATH with .singularity.d/libs → GLIBC_2.38/libdrm error).
# =============================================================================
apptainer run \
  --cleanenv \
  --writable-tmpfs \
  --env "ROBOT_PLATFORM=$ROBOT_PLATFORM" \
  --env "MUJOCO_GL=$MUJOCO_GL" \
  --env "PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM" \
  --env "EGL_PLATFORM=$EGL_PLATFORM" \
  --env "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES" \
  --env "NCCL_DEBUG=$NCCL_DEBUG" \
  --env "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM" \
  --env "TORCH_NCCL_BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT" \
  --env "NCCL_TIMEOUT=$NCCL_TIMEOUT" \
  --env "RAY_object_store_memory=$RAY_object_store_memory" \
  --env "RAY_memory_usage_threshold=$RAY_memory_usage_threshold" \
  --env "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" \
  --env "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING" \
  --env "TORCH_USE_CUDA_DSA=$TORCH_USE_CUDA_DSA" \
  --env "NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR" \
  --env "TRANSFORMERS_CACHE=$HF_CACHE_DIR" \
  --env "HF_HOME=$HF_CACHE_DIR" \
  --env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" \
  --env "HIP_VERSION=$HIP_VERSION_CONTAINER" \
  --env "HIP_PATH=$HIP_PATH_CONTAINER" \
  --env "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
  --env "WANDB_API_KEY=$WANDB_API_KEY" \
  --bind "$REPO_ROOT:$REPO_ROOT" \
  --bind "$NUMBA_CACHE_DIR:$NUMBA_CACHE_DIR" \
  --bind "/work1/chunyilee/yuhang:/work1/chunyilee/yuhang" \
  "$APPTAINER_IMAGE" \
  bash -c "$INNER_CMD"
