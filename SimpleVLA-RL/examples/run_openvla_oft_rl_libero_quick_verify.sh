#!/bin/bash
#SBATCH --job-name=grpo-libero-quick
#SBATCH --partition=staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=184
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/grpo_libero_quick_%j.out
#SBATCH --error=slurm/logs/grpo_libero_quick_%j.err
#SBATCH --nodelist=tn1-002-002

# Quick-verify run: train a few steps and validate often to check if GRPO is improving success rate.
# Uses the same Apptainer + config as run_openvla_oft_rl_libero_apptainer.sh, with overrides below.
#
# Usage (set SFT_MODEL_PATH and CKPT_PATH as for the full script):
#   export SFT_MODEL_PATH=/path/to/openvla-oft-sft-full
#   export CKPT_PATH=/path/to/checkpoints
#   export EXPERIMENT_NAME=grpo-libero-quick-verify   # optional, to avoid overwriting full run
#   sbatch examples/run_openvla_oft_rl_libero_quick_verify.sh
#
# What this does:
#   - TOTAL_EPOCHS=8   → train only 8 global steps (quick).
#   - TEST_FREQ=2      → validate every 2 steps → you get val/test_score/all at steps 0,2,4,6,8.
#   - SAVE_FREQ=4     → save checkpoints at steps 4 and 8.
# Check REPRODUCE_PAPER.md for how to interpret "has improved" (val/test_score/all should go up).
# Example: TRAINER_SAVE_FREQ=1 SFT_MODEL_PATH=checkpoints/openvla-oft-sft-full CKPT_PATH=checkpoints sbatch examples/run_openvla_oft_rl_libero_quick_verify.sh


export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-8}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-2}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-1}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-libero-quick-verify}"

# Use submit dir so we run the main script from the repo (not from Slurm's spool copy)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
exec bash "$REPO_ROOT/examples/run_openvla_oft_rl_libero_apptainer.sh"
