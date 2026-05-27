"""
finetune_real_world.py

Fine-tunes OpenVLA on a real-world robot dataset stored in HuggingFace
LeRobot format (parquet + mp4 videos). Designed for the SO-ARM-101 6-DOF
arm but configurable for any robot by setting ROBOT_PLATFORM appropriately.

Usage (single GPU):
    ROBOT_PLATFORM=SO101 python vla-scripts/finetune_real_world.py \
        --hf_repo_id christian0420/so101-poker-yellow-task \
        --vla_path openvla/openvla-7b \
        --run_root_dir ./runs_real_world

Usage (multi-GPU with torchrun):
    ROBOT_PLATFORM=SO101 torchrun --standalone --nnodes 1 --nproc-per-node 4 \
        vla-scripts/finetune_real_world.py \
        --hf_repo_id christian0420/so101-poker-yellow-task \
        --vla_path openvla/openvla-7b \
        --run_root_dir ./runs_real_world

IMPORTANT: ROBOT_PLATFORM=SO101 must be set in the environment (or as a prefix)
BEFORE this script runs so that prismatic.vla.constants picks up the right
ACTION_DIM=6, PROPRIO_DIM=6, NUM_ACTIONS_CHUNK=10 values at import time.
Alternatively, the env var is set programmatically below if not already present.
"""

import os

# Must be set before any prismatic import so constants.py reads the right values.
if "ROBOT_PLATFORM" not in os.environ:
    os.environ["ROBOT_PLATFORM"] = "SO101"

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
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
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
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
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.vla.datasets.hf_lerobot_dataset import LeRobotHFDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneRealWorldConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"            # HF Hub path or local path to base VLA
    processor_path: Optional[str] = None            # If set, load processor from here instead of vla_path.
                                                     # Needed when vla_path is a veRL/RL checkpoint that lacks
                                                     # processing_prismatic.py (set to the SFT base checkpoint).

    # Dataset
    hf_repo_id: str = "christian0420/so101-poker-yellow-task"  # HuggingFace dataset repo id
    task_instruction: str = "pick up the yellow poker chip"    # Fixed language instruction
    hf_cache_dir: Optional[str] = None                         # Local cache dir for HF downloads
    preload_resolution: int = 256                               # Resolution at which video frames are decoded and cached
    val_split: float = 0.1                                      # Fraction of episodes held out for validation

    run_root_dir: Path = Path("runs_real_world")               # Directory for logs and checkpoints

    # Architecture
    use_l1_regression: bool = True                             # Continuous action head with L1 loss (recommended)
    use_diffusion: bool = False                                # Diffusion action head (alternative to L1)
    num_diffusion_steps_train: int = 50
    use_film: bool = False                                     # FiLM language conditioning
    num_images_in_input: int = 2                               # 1 = top only, 2 = top + wrist
    use_proprio: bool = True                                   # Include joint state as proprio input

    # Training
    batch_size: int = 4                                        # Per-device batch size
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 500
    num_steps_before_decay: int = 30_000
    grad_accumulation_steps: int = 1
    max_steps: int = 50_000
    use_val_set: bool = True
    val_freq: int = 1_000
    val_time_limit: int = 60
    save_freq: int = 5_000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    diffusion_sample_freq: int = 50

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True

    # Logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "openvla-real-world"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10
    # fmt: on


# ---------------------------------------------------------------------------
# Utilities (shared with finetune.py)
# ---------------------------------------------------------------------------

def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def get_run_id(cfg: FinetuneRealWorldConfig) -> str:
    if cfg.run_id_override is not None:
        return cfg.run_id_override
    if cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
        return run_id
    dataset_tag = cfg.hf_repo_id.split("/")[-1]
    run_id = (
        f"{cfg.vla_path.split('/')[-1]}+{dataset_tag}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.image_aug:
        run_id += "--image_aug"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    ckpt_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {n}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneRealWorldConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)
    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused_params)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics = {}
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise = noisy_dict["noise"]
        noisy_actions = noisy_dict["noisy_actions"]
        diffusion_timestep_embeddings = noisy_dict["diffusion_timestep_embeddings"]
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

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
            zero_action_embeddings=True,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        metrics.update({
            "loss_value": loss.item(),
            "curr_action_accuracy": compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=current_action_mask).item(),
            "curr_action_l1_loss": compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask).item(),
            "next_actions_accuracy": compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask).item(),
            "next_actions_l1_loss": compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask).item(),
        })
    else:
        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        if use_l1_regression:
            predicted_actions = action_head.module.predict_action(actions_hidden_states)
            loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)

        if use_diffusion:
            noise_pred = action_head.module.predict_noise(actions_hidden_states).reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = _run_diffusion_sampling(
                        vla, action_head, noisy_action_projector, proprio_projector,
                        batch, batch_size, num_patches, ground_truth_actions.shape,
                        device_id, current_action_mask, next_actions_mask, use_proprio, use_film,
                    )

        metrics["loss_value"] = loss.item()

        should_log_l1 = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1:
            metrics["curr_action_l1_loss"] = torch.nn.L1Loss()(ground_truth_actions[:, 0], predicted_actions[:, 0]).item()
            metrics["next_actions_l1_loss"] = torch.nn.L1Loss()(ground_truth_actions[:, 1:], predicted_actions[:, 1:]).item()

    return loss, metrics


def _run_diffusion_sampling(
    vla, action_head, noisy_action_projector, proprio_projector,
    batch, batch_size, num_patches, actions_shape, device_id,
    current_action_mask, next_actions_mask, use_proprio, use_film,
) -> torch.Tensor:
    noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device_id, dtype=torch.bfloat16)
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)
    curr = noise
    for t in action_head.module.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        dte = action_head.module.time_encoder(timesteps).to(curr.dtype).to(curr.device).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=dte,
                use_film=use_film,
            )
            hs = out.hidden_states[-1][:, num_patches:-1]
            ahs = hs[current_action_mask | next_actions_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
            noise_pred = action_head.module.predict_noise(ahs)
        curr = action_head.module.noise_scheduler.step(noise_pred, t, curr).prev_sample
    return curr.reshape(actions_shape)


# ---------------------------------------------------------------------------
# Checkpoint, validation, logging
# ---------------------------------------------------------------------------

def _smoothen(deques: dict) -> dict:
    return {k: sum(dq) / len(dq) for k, dq in deques.items() if dq}


def _log_wandb(metrics: dict, prefix: str, step: int, run) -> None:
    run.log(
        {(f"{prefix}/Loss" if k == "loss_value" else f"{prefix}/{k.replace('_', ' ').title()}"): v
         for k, v in metrics.items()},
        step=step,
    )


def save_checkpoint(
    cfg, run_dir, log_step, vla, processor, proprio_projector,
    noisy_action_projector, action_head, train_dataset, distributed_state,
) -> None:
    if cfg.save_latest_checkpoint_only:
        ckpt_dir = run_dir
        suffix = "latest_checkpoint.pt"
    else:
        ckpt_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = ckpt_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, ckpt_dir)
        print(f"Saving checkpoint at step {log_step} → {ckpt_dir}")

    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(ckpt_dir)
        vla.module.save_pretrained(adapter_dir)
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), ckpt_dir / f"proprio_projector--{suffix}")
        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(noisy_action_projector.state_dict(), ckpt_dir / f"noisy_action_projector--{suffix}")
        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), ckpt_dir / f"action_head--{suffix}")
        if cfg.use_film:
            torch.save(vla.module.vision_backbone.state_dict(), ckpt_dir / f"vision_backbone--{suffix}")

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        merged = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
        if distributed_state.is_main_process:
            merged.save_pretrained(ckpt_dir)
            print(f"Saved merged model at step {log_step}")
        dist.barrier()


def run_validation(
    vla, action_head, noisy_action_projector, proprio_projector,
    val_dataloader, action_tokenizer, device_id, cfg, num_patches,
    log_step, distributed_state, val_time_limit,
) -> None:
    t0 = time.time()
    vla.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vla=vla, action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch, action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio, use_film=cfg.use_film,
                num_patches=num_patches, compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )
            metrics["loss"] = metrics["loss_value"]
            all_metrics.append(metrics)
            if time.time() - t0 > val_time_limit:
                break

    avg = {k: sum(m[k] for m in all_metrics if k in m) / len(all_metrics) for k in all_metrics[0]}
    avg["val_batches_count"] = len(all_metrics)
    if distributed_state.is_main_process:
        _log_wandb(avg, "VLA Val", log_step, wandb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def finetune(cfg: FinetuneRealWorldConfig) -> None:
    assert cfg.use_lora, "Only LoRA fine-tuning is supported."
    assert not (cfg.use_l1_regression and cfg.use_diffusion)

    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning {cfg.vla_path} on {cfg.hf_repo_id}")
    print(
        f"Constants: NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}, ACTION_DIM={ACTION_DIM}, "
        f"PROPRIO_DIM={PROPRIO_DIM}, NORM={ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    # Load model
    if model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    # processor_path lets RL checkpoints (which lack processing_prismatic.py) borrow
    # the processor from their SFT base checkpoint
    _proc_path = cfg.processor_path if cfg.processor_path else cfg.vla_path
    processor = AutoProcessor.from_pretrained(_proc_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout, target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    if cfg.use_film:
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone, llm_dim=vla.llm_dim,
        )
        if cfg.resume:
            vla.model.vision_backbone.load_state_dict(load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step))
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    vla = wrap_ddp(vla, device_id, find_unused=True)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector, "proprio_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    action_head = None
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead, "action_head", cfg, device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead, "action_head", cfg, device_id,
            {
                "input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM, "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim},
        )

    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        NUM_PATCHES += 1
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [p for p in noisy_action_projector.parameters() if p.requires_grad]
    if cfg.use_proprio:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Build datasets
    use_wrist = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist,
        use_proprio=cfg.use_proprio,
    )

    train_dataset = LeRobotHFDataset(
        repo_id=cfg.hf_repo_id,
        batch_transform=batch_transform,
        chunk_size=NUM_ACTIONS_CHUNK,
        task_instruction=cfg.task_instruction,
        cache_dir=cfg.hf_cache_dir,
        preload_resolution=cfg.preload_resolution,
        train=True,
        val_split=cfg.val_split,
    )

    if cfg.use_val_set:
        val_dataset = LeRobotHFDataset(
            repo_id=cfg.hf_repo_id,
            batch_transform=batch_transform,
            chunk_size=NUM_ACTIONS_CHUNK,
            task_instruction=cfg.task_instruction,
            cache_dir=cfg.hf_cache_dir,
            preload_resolution=cfg.preload_resolution,
            train=False,
            val_split=cfg.val_split,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=None,
        collate_fn=collator, num_workers=2, pin_memory=True,
        shuffle=True,
    )
    if cfg.use_val_set:
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, sampler=None,
            collate_fn=collator, num_workers=2, pin_memory=True,
        )

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Training loop
    # Unlike RLDSDataset (IterableDataset that loops forever), LeRobotHFDataset is a
    # regular Dataset that exhausts after one epoch. We wrap the dataloader in a
    # while-loop so training continues across epochs until max_steps is reached.
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        global_batch_idx = 0   # monotonically increasing across epochs
        training_done = False

        while not training_done:
            for batch in dataloader:
                batch_idx = global_batch_idx
                global_batch_idx += 1

                compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
                loss, metrics = run_forward_pass(
                    vla=vla, action_head=action_head,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch, action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_diffusion=cfg.use_diffusion,
                    use_proprio=cfg.use_proprio, use_film=cfg.use_film,
                    num_patches=NUM_PATCHES, compute_diffusion_l1=compute_diffusion_l1,
                    num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                )

                (loss / cfg.grad_accumulation_steps).backward()

                for k, v in metrics.items():
                    if k in recent_metrics:
                        recent_metrics[k].append(v)

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

                if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                    _log_wandb(_smoothen(recent_metrics), "VLA Train", log_step, wandb)

                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    for pg in optimizer.param_groups:
                        pg["lr"] = original_lr * (0.1 + 0.9 * lr_progress)

                if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                    wandb.log({"VLA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    grad_ok = all(torch.isfinite(p.grad).all() for p in trainable_params if p.grad is not None)
                    if not grad_ok:
                        if distributed_state.is_main_process:
                            print(f"[Grad NaN] step {log_step}, skipping optimizer step")
                            wandb.log({"VLA Train/Grad NaN Skipped": 1}, step=log_step)
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    progress.update()

                if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                    save_checkpoint(
                        cfg=cfg, run_dir=run_dir, log_step=log_step, vla=vla,
                        processor=processor, proprio_projector=proprio_projector if cfg.use_proprio else None,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                        train_dataset=train_dataset, distributed_state=distributed_state,
                    )

                if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                    run_validation(
                        vla=vla, action_head=action_head,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        val_dataloader=val_dataloader, action_tokenizer=action_tokenizer,
                        device_id=device_id, cfg=cfg, num_patches=NUM_PATCHES,
                        log_step=log_step, distributed_state=distributed_state,
                        val_time_limit=cfg.val_time_limit,
                    )
                    vla.train()

                if log_step >= cfg.max_steps:
                    print(f"Reached max_steps={cfg.max_steps}. Stopping.")
                    training_done = True
                    break


if __name__ == "__main__":
    finetune()
