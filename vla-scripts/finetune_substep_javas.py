"""
finetune_substep.py

Fine-tunes OpenVLA via LoRA with per-timestep substep instructions.

This is an extended version of finetune.py that supports replacing per-episode task instructions
with per-timestep substep instructions from a JSON label file. This enables training with more
fine-grained language supervision at each timestep.

Usage:
    python vla-scripts/finetune_substep.py \
        --vla_path openvla/openvla-7b \
        --data_root_dir datasets/rlds \
        --dataset_name libero_goal_no_noops \
        --substep_labels_path substep_labels_output.json \
        --run_root_dir runs \
        --wandb_entity your-entity \
        --wandb_project your-project
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
# Import substep-aware dataset classes
from prismatic.vla.datasets.datasets_substep import SubstepRLDSBatchTransform, SubstepRLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Import forward pass and validation functions from original finetune script
# Note: These functions are imported from the original finetune.py in the same directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from finetune import (
    remove_ddp_in_checkpoint,
    load_checkpoint,
    wrap_ddp,
    count_parameters,
    init_module,
    run_diffusion_sampling,
    compute_smoothened_metrics,
    log_metrics_to_wandb,
    save_training_checkpoint,
    run_validation,
)
# Note: We define our own run_forward_pass with EOS weighting below

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    eos_head=None,              # EOS classification head (optional)
    lambda_eos=1.0,             # EOS loss weight
    use_global_weights=True,    # Use global fixed weights
    pos_weight=50.0,            # Positive class weight for global weighting
    use_focal_loss=False,       # Use Focal Loss instead of BCE
    focal_alpha=0.25,           # Focal Loss alpha
    focal_gamma=2.0,            # Focal Loss gamma
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass with optional EOS classification.
    
    This version supports:
    - Standard action prediction (L1 regression or diffusion)
    - Optional separate EOS classification head with batch accumulation balancing
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)

        if use_l1_regression:
            # Predict action
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            # Get full L1 loss
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        # Prepare metrics dict
        metrics_dict = {"loss_value": loss.item()}
        metrics.update(metrics_dict)

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # [EOS CLASSIFICATION] Compute EOS loss with weighted BCE or Focal Loss
    # 关键改进：
    # 1. 保持梯度流从loss回传到VLA主干，实现真正的Fine-tuning
    # 2. 使用全局固定权重处理极端不平衡（如1:800）
    # 3. 不跳过任何batch，包括全是负样本的batch
    eos_loss = torch.tensor(0.0, device=device_id)
    
    if eos_head is not None:
        # EOS head only works with L1 regression or diffusion (requires actions_hidden_states)
        if use_l1_regression or use_diffusion:
            # Get ground truth EOS labels from batch
            if "eos_labels" in batch:
                eos_gt = batch["eos_labels"].to(device_id).to(torch.float32)  # (B, NUM_ACTIONS_CHUNK, 1)
                
                # Forward through EOS head (保持梯度！)
                # actions_hidden_states: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim) - 有梯度连接到VLA
                eos_logits = eos_head.module.forward(actions_hidden_states)  # (B, NUM_ACTIONS_CHUNK, 1)
                eos_logits = torch.clamp(eos_logits, min=-20.0, max=20.0)

                # Flatten for loss computation
                eos_logits_flat = eos_logits.squeeze(-1).reshape(-1)  # (B * NUM_ACTIONS_CHUNK,)
                eos_gt_flat = eos_gt.squeeze(-1).reshape(-1)          # (B * NUM_ACTIONS_CHUNK,)
                
                # Count positive/negative samples
                num_pos = (eos_gt_flat > 0.5).sum().item()
                num_neg = (eos_gt_flat <= 0.5).sum().item()
                total_samples = num_pos + num_neg
                
                # 计算loss（不再跳过任何batch！）
                if use_focal_loss:
                    # Focal Loss: 专门为极端不平衡设计
                    # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
                    eos_probs = torch.sigmoid(eos_logits_flat)
                    
                    # Compute focal weight: (1 - p_t)^gamma
                    # For positive samples: p_t = p, for negative: p_t = 1 - p
                    p_t = torch.where(eos_gt_flat > 0.5, eos_probs, 1 - eos_probs)
                    focal_weight = (1 - p_t) ** focal_gamma
                    
                    # Compute alpha weight (class balance)
                    alpha_t = torch.where(
                        eos_gt_flat > 0.5,
                        torch.tensor(focal_alpha, device=device_id),
                        torch.tensor(1 - focal_alpha, device=device_id)
                    )
                    
                    # BCE loss
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(
                        eos_logits_flat, eos_gt_flat, reduction='none'
                    )
                    
                    # Focal loss = alpha * focal_weight * BCE
                    eos_loss = (alpha_t * focal_weight * bce_loss).mean()
                    
                    weight_info = {
                        'eos_focal_alpha': focal_alpha,
                        'eos_focal_gamma': focal_gamma,
                    }
                else:
                    # Weighted BCE Loss
                    if use_global_weights:
                        # 使用全局固定权重（适合极端不平衡）
                        # pos_weight: 正样本的权重倍数
                        # 负样本权重固定为1.0，正样本权重为pos_weight
                        pos_weight_tensor = torch.tensor([pos_weight], device=device_id)
                        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                        eos_loss = loss_fn(eos_logits_flat, eos_gt_flat)
                        
                        weight_info = {
                            'eos_pos_weight': pos_weight,
                            'eos_neg_weight': 1.0,
                            'eos_weight_type': 'global_fixed',
                        }
                    else:
                        # 动态权重（batch内平衡，不适合极端不平衡）
                        if num_pos > 0 and num_neg > 0:
                            # 让正负样本的total loss贡献相等
                            weight_pos = total_samples / (2.0 * num_pos)
                            weight_neg = total_samples / (2.0 * num_neg)
                        elif num_pos > 0:
                            # 全是正样本（极少见）
                            weight_pos = 1.0
                            weight_neg = 1.0
                        else:
                            # 全是负样本（常见）
                            weight_pos = 1.0
                            weight_neg = 1.0
                        
                        # Create sample weights
                        sample_weights = torch.where(
                            eos_gt_flat > 0.5,
                            torch.tensor(weight_pos, device=device_id),
                            torch.tensor(weight_neg, device=device_id)
                        )
                        
                        # Weighted BCE loss
                        loss_fn = nn.BCEWithLogitsLoss(weight=sample_weights)
                        eos_loss = loss_fn(eos_logits_flat, eos_gt_flat)
                        
                        weight_info = {
                            'eos_pos_weight': weight_pos,
                            'eos_neg_weight': weight_neg,
                            'eos_weight_type': 'dynamic',
                        }
                
                # Compute classification metrics
                with torch.no_grad():
                    from prismatic.models.action_heads import compute_classification_metrics
                    eos_pred = torch.sigmoid(eos_logits_flat) > 0.5
                    eos_metrics = compute_classification_metrics(
                        eos_pred, eos_gt_flat > 0.5
                    )
                    eos_metrics.update({
                        'eos_num_positive': num_pos,
                        'eos_num_negative': num_neg,
                        'eos_ratio': num_pos / max(1, total_samples),
                        **weight_info,
                    })
                    metrics.update(eos_metrics)
            else:
                # No EOS labels in batch (shouldn't happen with SubstepRLDSDataset)
                metrics.update({'eos_no_labels': 1.0})
    
    # Combine action loss and EOS loss
    action_loss = loss  # Rename for clarity
    total_loss = action_loss + lambda_eos * eos_loss
    
    # Update metrics
    metrics.update({
        'action_loss': action_loss.item(),
        'eos_loss': eos_loss.item(),
        'total_loss': total_loss.item(),
    })
    
    # Return both the total loss tensor (with gradients) and the metrics dictionary (with detached values)
    return total_loss, metrics


@dataclass
class FinetuneSubstepConfig:
    """
    Configuration for fine-tuning OpenVLA with per-timestep substep instructions.
    
    Extends FinetuneConfig with substep_labels_path parameter.
    """
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "libero_goal_no_noops"       # Name of fine-tuning dataset
    substep_labels_path: str = "substep_labels_output.json"  # Path to substep labels JSON file
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    max_grad_norm: float = 0.8                       # Maximum gradient norm for clipping (防止梯度爆炸，特别是使用高pos_weight时)
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # Substep EOS training
    use_substep_eos: bool = True                     # If True, inserts EOS token at substep boundaries for training
    
    # EOS Classification (separate head with weighted BCE loss)
    use_eos_classification: bool = True              # If True, uses separate EOS classification head
    eos_hidden_dim: int = 1024                       # Hidden dimension for EOS classification head
    eos_dropout: float = 0.1                         # Dropout rate for EOS classification head
    lambda_eos: float = 1.0                          # Weight for EOS loss in total loss
    eos_threshold: float = 0.5                       # Threshold for EOS detection during inference
    
    # EOS class weights (for extreme imbalance like 1:800)
    eos_use_global_weights: bool = True              # Use global fixed weights instead of per-batch dynamic weights
    eos_pos_weight: float = 20.0                     # Fixed weight for EOS=1 class (针对极端不平衡，如1:800，设为50-100)
    eos_use_focal_loss: bool = True                 # Use Focal Loss (better for extreme imbalance)
    eos_focal_alpha: float = 0.25                    # Focal Loss alpha (weight for positive class)
    eos_focal_gamma: float = 2.0                     # Focal Loss gamma (focusing parameter)

    # fmt: on


def get_run_id(cfg: FinetuneSubstepConfig) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneSubstepConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        # Add substep indicator to run ID
        run_id += "--substep"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


@draccus.wrap()
def finetune_substep(cfg: FinetuneSubstepConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA with per-timestep substep instructions.

    This function extends the original finetune() by using SubstepRLDSDataset which replaces
    per-episode task instructions with per-timestep substep instructions from a JSON file.

    Args:
        cfg (FinetuneSubstepConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )
    
    # Validate substep labels path exists
    if not os.path.exists(cfg.substep_labels_path):
        raise FileNotFoundError(f"Substep labels file not found: {cfg.substep_labels_path}")

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}` with substep instructions")
    print(f"Substep labels path: {cfg.substep_labels_path}")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path

    # Register OpenVLA model to HF Auto Classes using LOCAL class definitions.
    # This ensures we always use the local code (including PrismaticProcessor) regardless
    # of what version is cached in the checkpoint directory or HF module cache.
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA using locally registered classes (trust_remote_code=False),
    # which avoids loading potentially stale processing_prismatic.py from the checkpoint dir.
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA setup
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # If applicable, instantiate EOS classification head
    eos_head = None
    if cfg.use_eos_classification:
        print(f"\n{'='*80}")
        print(f"[EOS Classification] Initializing separate EOS head")
        print(f"  Strategy: Weighted BCE loss with gradient flow to VLA backbone")
        print(f"  Loss type: {'Focal Loss' if cfg.eos_use_focal_loss else 'Weighted BCE'}")
        if cfg.eos_use_focal_loss:
            print(f"  Focal Loss: alpha={cfg.eos_focal_alpha}, gamma={cfg.eos_focal_gamma}")
        else:
            if cfg.eos_use_global_weights:
                print(f"  Global weights: pos={cfg.eos_pos_weight}, neg=1.0 (fixed)")
            else:
                print(f"  Dynamic weights: computed per-batch")
        print(f"  Lambda EOS: {cfg.lambda_eos}")
        print(f"  Gradient clipping: {'Enabled' if cfg.max_grad_norm > 0 else 'Disabled'}")
        if cfg.max_grad_norm > 0:
            print(f"    Max grad norm: {cfg.max_grad_norm}")
        print(f"  ✓ 真正的Fine-tuning（梯度回传到VLA主干）")
        print(f"{'='*80}\n")
        
        from prismatic.models.action_heads import EOSClassificationHead
        
        # Initialize EOS classification head
        eos_head = init_module(
            EOSClassificationHead,
            "eos_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": cfg.eos_hidden_dim,
                "dropout": cfg.eos_dropout,
            },
            to_bf16=True,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    if cfg.use_eos_classification:
        trainable_params += [param for param in eos_head.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # [CRITICAL] Create substep-aware batch transform and dataset
    # This is the key difference from the original finetune.py
    use_wrist_image = cfg.num_images_in_input > 1
    
    batch_transform = SubstepRLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        substep_labels={},  # Will be loaded by SubstepRLDSDataset
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_substep_eos=cfg.use_substep_eos,  # Enable EOS token insertion at substep boundaries
    )
    
    train_dataset = SubstepRLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        cfg.substep_labels_path,  # Pass substep labels path
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    if cfg.use_val_set:
        val_dataset = SubstepRLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            cfg.substep_labels_path,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        # EOS classification metrics
        "action_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "total_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_precision": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_recall": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_f1": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_num_positive": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_num_negative": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_ratio": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_pos_weight": deque(maxlen=cfg.grad_accumulation_steps),
        "eos_neg_weight": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # [EOS DEBUG] Sample batches to check EOS distribution before training
    if cfg.use_eos_classification and distributed_state.is_main_process:
        print(f"\n{'='*80}")
        print(f"[EOS DEBUG] Checking EOS distribution in dataset...")
        print(f"  Sampling first 50 batches to estimate EOS=1 ratio...")
        
        sample_eos_positive = 0
        sample_eos_negative = 0
        sample_batches = 0
        
        try:
            # Use the same dataloader (don't create a new one)
            temp_iter = iter(train_dataset)

            for sample_idx in range(50):  # Sample 50 batches
                try:
                    sample_batch = next(temp_iter)
                    
                    # Check if eos_labels exist
                    if "eos_labels" in sample_batch:
                        eos_gt = sample_batch["eos_labels"]
                        # Convert to numpy if it's a tensor
                        if isinstance(eos_gt, torch.Tensor):
                            eos_gt = eos_gt.numpy()
                        elif not isinstance(eos_gt, np.ndarray):
                            eos_gt = np.array(eos_gt)
                        
                        sample_eos_positive += (eos_gt > 0.5).sum()
                        sample_eos_negative += (eos_gt <= 0.5).sum()
                        sample_batches += 1
                    else:
                        print(f"  ⚠️  WARNING: Batch {sample_idx} has no 'eos_labels' field!")
                        break
                        
                except StopIteration:
                    print(f"  Dataset exhausted after {sample_batches} batches")
                    break
            
            if sample_batches > 0:
                total_samples = sample_eos_positive + sample_eos_negative
                eos_ratio = sample_eos_positive / total_samples if total_samples > 0 else 0
                avg_pos_per_batch = sample_eos_positive / sample_batches
                
                print(f"  Sampled {sample_batches} batches, {total_samples} total action samples")
                print(f"  EOS=1: {sample_eos_positive} ({eos_ratio*100:.2f}%)")
                print(f"  EOS=0: {sample_eos_negative} ({(1-eos_ratio)*100:.2f}%)")
                print(f"  Average EOS=1 per batch: {avg_pos_per_batch:.2f}")
                
                print(f"  Global imbalance ratio: 1:{int(1.0/eos_ratio)}")
                
                # 给出权重建议
                if cfg.eos_use_global_weights:
                    # 计算推荐的pos_weight
                    # 一般设为 neg_samples / pos_samples，但可以适当调整
                    recommended_weight = (1 - eos_ratio) / eos_ratio if eos_ratio > 0 else 100
                    recommended_weight = min(recommended_weight, 100)  # 上限100
                    
                    print(f"\n  [Weight Recommendation]")
                    print(f"    Current eos_pos_weight: {cfg.eos_pos_weight:.1f}")
                    print(f"    Recommended eos_pos_weight: {recommended_weight:.1f}")
                    if abs(cfg.eos_pos_weight - recommended_weight) > 20:
                        print(f"    ⚠️  Consider adjusting: --eos_pos_weight={recommended_weight:.0f}")
                
                if eos_ratio < 0.01:
                    print(f"\n  ⚠️  EXTREME IMBALANCE DETECTED ({eos_ratio*100:.4f}%)!")
                    print(f"  Recommendations:")
                    print(f"    1. 使用全局固定权重: --eos_use_global_weights=True")
                    print(f"    2. 设置高权重: --eos_pos_weight={int(recommended_weight)}")
                    print(f"    3. 或使用Focal Loss: --eos_use_focal_loss=True")
                    print(f"    4. 增加lambda_eos: --lambda_eos=2.0")
                elif eos_ratio < 0.1:
                    print(f"\n  ⚠️  High imbalance ({eos_ratio*100:.2f}%)")
                    print(f"  建议使用全局固定权重: --eos_pos_weight={int(recommended_weight)}")
            else:
                print(f"  ⚠️  ERROR: Could not sample any batches from dataset!")
                print(f"  Check that SubstepRLDSDataset is properly configured")
                
        except Exception as e:
            print(f"  ⚠️  ERROR during sampling: {e}")
            print(f"  Skipping EOS distribution check...")
        
        print(f"{'='*80}\n")
    
    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):

            # Compute training metrics and loss
            compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                eos_head=eos_head if cfg.use_eos_classification else None,
                lambda_eos=cfg.lambda_eos,
                use_global_weights=cfg.eos_use_global_weights,
                pos_weight=cfg.eos_pos_weight,
                use_focal_loss=cfg.eos_use_focal_loss,
                focal_alpha=cfg.eos_focal_alpha,
                focal_gamma=cfg.eos_focal_gamma,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # [SAFETY] Check for NaN/Inf in loss (critical for high pos_weight training)
            if not torch.isfinite(normalized_loss):
                print(f"❌ [NaN/Inf Error] Step {batch_idx}: loss={normalized_loss.item()}")
                print(f"   Metrics: {metrics}")
                print(f"   Skipping this batch to prevent training crash...")
                optimizer.zero_grad()  # Clear any accumulated gradients
                continue  # Skip this batch
            
            # Backward pass
            normalized_loss.backward()

            # [DEBUG] Print EOS loss every batch (first 100 batches)
            if distributed_state.is_main_process and batch_idx < 100 and cfg.use_eos_classification:
                print(f"[Batch {batch_idx}] eos_loss={metrics.get('eos_loss', 0):.4f}, "
                      f"action_loss={metrics.get('action_loss', 0):.4f}, "
                      f"total_loss={metrics.get('total_loss', 0):.4f}")
            
            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                
                # [EOS DEBUG] Print EOS stats every 50 steps
                if cfg.use_eos_classification and log_step % 50 == 0:
                    avg_num_pos = smoothened_metrics.get('eos_num_positive', 0)
                    avg_num_neg = smoothened_metrics.get('eos_num_negative', 0)
                    avg_ratio = smoothened_metrics.get('eos_ratio', 0)
                    avg_pos_weight = smoothened_metrics.get('eos_pos_weight', 0)
                    avg_neg_weight = smoothened_metrics.get('eos_neg_weight', 0)
                    avg_eos_loss = smoothened_metrics.get('eos_loss', 0)
                    avg_eos_acc = smoothened_metrics.get('eos_accuracy', 0)
                    avg_eos_recall = smoothened_metrics.get('eos_recall', 0)
                    avg_eos_precision = smoothened_metrics.get('eos_precision', 0)
                    
                    print(f"\n{'='*80}")
                    print(f"[EOS Stats] Step {log_step}")
                    print(f"  Batch samples: {avg_num_pos:.1f} EOS=1, {avg_num_neg:.1f} EOS=0 (ratio: {avg_ratio:.4f})")
                    if cfg.eos_use_global_weights:
                        print(f"  Global weights: pos={cfg.eos_pos_weight:.1f}, neg=1.0 (固定)")
                    else:
                        print(f"  Dynamic weights: pos={avg_pos_weight:.2f}, neg={avg_neg_weight:.2f}")
                    print(f"  Loss: {avg_eos_loss:.4f}")
                    print(f"  Metrics: Acc={avg_eos_acc:.3f}, Recall={avg_eos_recall:.3f}, Prec={avg_eos_precision:.3f}")
                    print(f"  ✓ 每个batch都更新，梯度流向VLA主干")
                    print(f"{'='*80}\n")

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # [CRITICAL] Gradient clipping to prevent explosion (especially with high pos_weight)
                # Clip gradients of all trainable parameters
                if cfg.max_grad_norm > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_params, 
                        max_norm=cfg.max_grad_norm
                    )
                    # Log gradient norm for monitoring
                    if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                        wandb.log({"VLA Train/Grad Norm": total_norm.item()}, step=log_step)
                        # Warn if gradient norm is very large (before clipping)
                        if total_norm.item() > cfg.max_grad_norm * 10:
                            print(f"⚠️  [Grad Norm Warning] Step {log_step}: grad_norm={total_norm.item():.2f} "
                                  f"(clipped to {cfg.max_grad_norm})")
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                # Save standard components
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )
                
                # [SUBSTEP EOS] Save EOS head separately
                if cfg.use_eos_classification and eos_head is not None and distributed_state.is_main_process:
                    if cfg.save_latest_checkpoint_only:
                        checkpoint_dir = run_dir
                        checkpoint_name_suffix = "latest_checkpoint.pt"
                    else:
                        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
                        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"
                    
                    eos_head_path = checkpoint_dir / f"eos_head--{checkpoint_name_suffix}"
                    torch.save(eos_head.state_dict(), eos_head_path)
                    print(f"✓ Saved EOS head checkpoint: {eos_head_path}")
                
                # Wait for EOS head to be saved
                dist.barrier()

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    set_seed(42) # reproducibility
    finetune_substep()

