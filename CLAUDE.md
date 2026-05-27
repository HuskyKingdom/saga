# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenVLA-OFT is a Vision-Language-Action (VLA) model fine-tuning framework for robot learning. It fine-tunes pretrained OpenVLA models via LoRA for simulation (LIBERO) and real-world robot tasks (ALOHA). Key capabilities: L1 regression or diffusion-based action heads, optional FiLM language grounding, proprioceptive state integration, and action chunking.

## Setup

```bash
conda create -n openvla-oft python=3.10 -y && conda activate openvla-oft
pip3 install torch torchvision torchaudio
pip install -e .
pip install packaging ninja && pip install "flash-attn==2.5.5" --no-build-isolation

# For LIBERO evaluation
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Common Commands

**Fine-tuning (multi-GPU):**
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /YOUR/CHECKPOINTS/DIR/ \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 2 --use_proprio True \
  --batch_size 8 --learning_rate 5e-4 \
  --num_steps_before_decay 100000 --max_steps 150005 \
  --save_freq 10000 --image_aug True --lora_rank 32 \
  --wandb_entity "ENTITY" --wandb_project "PROJECT"
```

**LIBERO evaluation:**
```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True
```

**Merge LoRA weights offline:**
```bash
python vla-scripts/merge_lora_weights_and_save.py --checkpoint /path/to/lora/checkpoint
```

**Linting/formatting:**
```bash
black --line-length 121 <file>
ruff check --fix <file>
```

## Server Infrastructure

Two remote servers are used for this project. Login with `ssh yuhang@<host>`, then run `work` to activate the environment and navigate to the workspace.

### AMD Server — Training (`hpcfund-tnn.amd.com`)
- `work` alias: `cd /work1/chunyilee/yuhang/openvla-oft-yhs && conda activate vla`
- Workspace: `/work1/chunyilee/yuhang/openvla-oft-yhs`
- Data: `/work1/chunyilee/yuhang/modified_libero_rlds`
- Checkpoints: `/work1/chunyilee/yuhang/openvla-oft-yhs/ckpoints`
- HF cache: `HF_HOME=/work1/chunyilee/yuhang/`
- **AMD-specific training flag** (required): `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` must be set before `torchrun`
- SLURM partitions: `mi3508xl` (8× GPU), `vlm` (4× GPU)
- Submit jobs: `sbatch slurms/train_infobot.sh` or `sbatch slurms/training_amd.sh`

### NV Server — Evaluation (`140.114.89.63`)
- `work` alias: `conda activate vla && cd ~/Warehouse/Yuhangworkspace/openvla-oft-yhs`
- Workspace: `~/Warehouse/Yuhangworkspace/openvla-oft-yhs`
- **Before any evaluation**, export LIBERO-PRO to PYTHONPATH:
  ```bash
  export PYTHONPATH=$PYTHONPATH:/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO
  ```
- Evaluations are run inside a `screen` session named **`exp`**: `screen -S exp` to create, `screen -r exp` to reattach
- Reference eval script: `auto_eval_nv40_pro.sh` — runs all 4 LIBERO-PRO task suites × 4 generalization modes (language, task, swap, object), 50 trials per task, using `run_libero_pro_eval_substep.py`
- Checkpoint naming convention on this server:
  `ckpt/ckpoints/openvla-7b+<dataset>+b<bs>+lr-<lr>+lora-r<rank>+dropout-0.0--image_aug--<run_id>--<step>_chkpt`

## LIBERO-PRO Evaluation

The PRO benchmark tests generalization beyond standard LIBERO. Evaluation config lives in `experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml` and is patched via `sed` before each run to switch the active mode.

**Full automated eval (from NV server workspace):**
```bash
# Set PYTHONPATH first (see NV server section above)
bash auto_eval_nv40_pro.sh
```

**Single eval run (example):**
```bash
python experiments/robot/libero/run_libero_pro_eval_substep.py \
  --pretrained_checkpoint ckpt/ckpoints/<checkpoint_name> \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml \
  --unnorm_key libero_spatial \
  --task_label my_label_spatial_lan \
  --use_eos_detection False \
  --use_proprio True \
  --use_l1_regression True \
  --use_bddl_language True \
  --num_images_in_input 2
```

Reset the evaluation config flags before switching modes:
```bash
bash experiments/robot/libero/LIBERO-PRO/reset_eval_config.sh \
  experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
```

## SimpleVLA-RL (GRPO Reinforcement Learning)

Located in `SimpleVLA-RL/`. Uses [veRL](https://github.com/volcengine/verl) (v0.2 or v0.3) for GRPO-based RL on top of a fine-tuned SFT checkpoint. Runs on the AMD server via SLURM.

**Key paths (AMD server):**
- SFT starting checkpoint: `/work1/chunyilee/yuhang/openvla-oft-yhs/landmarked_ckpoints/apd_discrete_160k`
- APD plans file: `/work1/chunyilee/yuhang/openvla-oft-yhs/APD_plans_scaled.json`

**Launch RL training (`apd_trail.sh` on AMD server):**
```bash
# Set variables, then submit
export SFT_MODEL_PATH="/work1/chunyilee/yuhang/openvla-oft-yhs/landmarked_ckpoints/apd_discrete_160k"
export CKPT_PATH="./exp_out"
export APD_PLANS_PATH="/work1/chunyilee/yuhang/openvla-oft-yhs/APD_plans_scaled.json"
export CONTRASTIVE_REWARD_COEF=2
export DATASET_NAME="libero_4_task_suites"
export NUM_GPUS=8
sbatch examples/run_openvla_oft_substep_rl_libero.sh
```

**Quick verify (few-step smoke test):**
```bash
sbatch examples/run_openvla_oft_rl_libero_quick_verify.sh  # 8 training steps, validates every 2
```

**SimpleVLA-RL setup** (separate conda env `simplevla`, see `SimpleVLA-RL/SETUP.md`):
- Requires veRL, OpenVLA-OFT, and LIBERO installed at the same directory level
- For RoboTwin 2.0 support, additional patching via `copy_overwrite_robotwin2.sh` is needed

## Code Style

- Black line length: 121 characters
- Ruff rules: A, B, E, F, I, RUF, W (F722 ignored; F401/E402 ignored in `__init__.py`)
- Pre-commit hooks enforce these automatically

## Architecture

### Core Package: `prismatic/`

**`prismatic/vla/`** — VLA-specific logic
- `action_tokenizer.py` — discretizes continuous actions into tokens (legacy; OFT uses continuous heads)
- `constants.py` — per-robot constants: action chunk size, action dim, proprio dim (LIBERO=8/7/8, ALOHA=25/14/14, BRIDGE=5/7/7)
- `datasets/rlds/` — RLDS format dataset loaders and batch transforms

**`prismatic/models/`** — Model components
- `action_heads.py` — `L1RegressionActionHead`, `DiffusionActionHead`; produce action chunks from LLM hidden states
- `film_vit_wrapper.py` — FiLM conditioning wrapper over ViT backbone for language grounding
- `projectors.py` — `ProprioProjector` (proprio → embedding), `NoisyActionProjector`
- `vlas/openvla.py` — `OpenVLAForActionPrediction`: main model class combining ViT + LLM + action head
- `backbones/` — ViT vision backbones and LLM backbones (e.g., Llama-2)
- `materialize.py` — instantiates components from config

**`prismatic/training/`**
- `train_utils.py` — loss computation, action masking, gradient clipping
- `strategies/` — distributed training strategies (FSDP, DDP)

**`prismatic/extern/hf/`** — HuggingFace-compatible config, model, and processor wrappers (enables `from_pretrained`)

### Training Scripts: `vla-scripts/`
- `finetune.py` — main LoRA fine-tuning entry point; loads pretrained model, applies LoRA via PEFT, trains on RLDS data
- `finetune_substep.py` — substep-decomposed fine-tuning variant
- `deploy.py` — launches VLA server for real-robot (ALOHA) inference via network
- `hnn_utils.py` — hierarchical NN utilities for substep approaches

### Evaluation: `experiments/robot/`
- `libero/run_libero_eval.py` — LIBERO simulation eval; rolls out policy in LIBERO envs, computes success rate
- `aloha/run_aloha_eval.py` — ALOHA real-robot eval client (pairs with `deploy.py` server)
- `openvla_utils.py` — shared checkpoint loading and action generation utilities
- `robot_utils.py` — shared action normalization and environment interaction

### Data Flow

1. RLDS datasets → `RLDSDataset` (episodes) → `RLDSBatchTransform` (augmentation) → collator (batching)
2. Batch: images + language instruction + proprio state → ViT backbone → projection → LLM → action head → action chunk (continuous)
3. Loss: L1 on predicted vs. ground-truth actions (or diffusion denoising loss)
4. Evaluation: load checkpoint → process observations → generate action chunk → step environment → collect success metrics

### Key Design Patterns
- **LoRA via PEFT:** All fine-tuning uses LoRA adapters; `lora_rank=32` is default. Weights can be merged offline via `merge_lora_weights_and_save.py`.
- **Action chunking:** Model predicts multiple timesteps at once (chunk size varies by robot platform, defined in `constants.py`).
- **Multi-image input:** `num_images_in_input` controls how many camera views are concatenated (typically 2 for wrist + third-person).
- **Proprio integration:** `use_proprio=True` feeds robot state through `ProprioProjector` into the LLM token sequence.
- **HF compatibility:** Models expose standard HF interfaces; checkpoints are distributed via HuggingFace Hub.

### Notable Subdirectories
- `docs/` — internal technical notes on EOS handling, 7D→8D action modifications, OFT action generation
- `slurms/` — SLURM job templates for cluster training
- `SimpleVLA-RL/` — RL fine-tuning variant (separate setup)
- `Javas/` — research notes on substep design
- `logs/` — daily experiment logs (YYYY-MM-DD.md), maintained per session

## Research Goal

Solve **instruction-ignorance** in VLA models. Theoretical basis: `Information-Theoretic Analysis of Instruction Grounding in VLA Models.pdf`.

**Core problem:** Standard VLA training → H(L|V) ≈ 0 → I(A;L|V) ≈ 0 → language gradient vanishes → model degenerates to vision-only policy.

**Two-stage fix (APD + Contrastive RL):**
1. **APD SFT** — Replace fixed task instructions with time-varying substep instructions. H(L^sub|V) >> 0 (same visual state, different substep → different action needed). Language gradient reactivated.
2. **Contrastive RL (GRPO)** — Explicitly maximise I(A;L|V): reward `r = min(||a+ - a-||_2/sqrt(d), 1)` where a+ is action under correct substep, a- is action under a wrong instruction from the same suite.

**Success criteria (LIBERO-PRO benchmark):**
- `task` mode (instruction swap): **>45% SR** for most task suites
- `sem`/`object` modes: **>85% SR** (not required to be SOTA, but must not be too low)

## Model Variants

| Model | L1 | Images | Proprio | Purpose |
|-------|----|--------|---------|---------|
| OFT | ✓ | 2 | ✓ | Baseline |
| APD_scalled | ✓ | 2 | ✓ | Main APD SFT — script: `slurms/substep_plus_scalled.sh` |
| APD_discrete | ✗ | 1 | ✗ | RL-compatible SFT base — script: `slurms/substep_plus_scalled_regressive.sh` |
| RL model | ✗ | 1 | ✗ | GRPO on APD_discrete — script: `SimpleVLA-RL/apd_trail.sh` |

APD_discrete (no L1, no proprio, 1 image) is required as RL SFT base because SimpleVLA-RL uses autoregressive vLLM generation, which is incompatible with the L1 regression head.

## Experiment Workflow Loop

1. **Check status**: `squeue -u yuhang` on AMD; `screen -r exp` on NV
2. **If running**: wait — do not queue new jobs
3. **Transfer checkpoint AMD → NV**:
   ```bash
   # AMD: tar and scp
   tar -czf <ckpt>.tar.gz ckpoints/<ckpt>/
   scp <ckpt>.tar.gz yuhang@140.114.89.63:~/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/
   # NV: extract
   tar -xzf ckpt/ckpoints/<ckpt>.tar.gz -C ckpt/ckpoints/
   ```
4. **Evaluate** (NV, inside `screen -S exp` / `screen -r exp`):
   ```bash
   export PYTHONPATH=$PYTHONPATH:/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO
   # Edit PRETRAINED_CHECKPOINT in auto_eval_nv40_pro.sh, then:
   bash auto_eval_nv40_pro.sh
   ```
5. **Code changes + push**:
   ```bash
   git add <files>
   git commit -m "Veldt- <brief description>"   # always Veldt- prefix
   git push
   # AMD: git pull && sbatch slurms/<script>.sh
   ```
6. **Daily log**: update `logs/YYYY-MM-DD.md`; recap at end of each day

## Current Experiment State (as of 2026-04-07)

- **AMD job #15417** running: GRPO RL on `apd_discrete_160k`, `oft-substep-rl-libero`, ~33h left
  - Step 3/100, val score 0.125, contrastive reward 0.206
- **NV eval**: `oft_plus--150000_chkpt` running in `screen exp`
- **Best task-mode result so far**: APD_scalled spatial 31.4% (target: >45%)
- **Next action**: wait for both to finish, evaluate RL checkpoint
