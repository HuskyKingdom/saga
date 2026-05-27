"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Apply JSON numpy patch for serialization
json_numpy.patch()

# Global flag for FLOPs calculation (only compute once)
_FLOPS_CALCULATED = False

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType


def normalize_gripper_action_tensor(action: torch.Tensor, binarize: bool = True) -> torch.Tensor:
    """
    Normalize gripper action from [0,1] to [-1,+1] range (PyTorch version).

    Args:
        action (torch.Tensor): Action tensor with gripper action in the last dimension
        binarize (bool): Whether to binarize gripper action to -1 or +1

    Returns:
        torch.Tensor: Action tensor with normalized gripper action
    """
    # 复制，避免修改原 tensor
    normalized_action = action.clone()

    # Normalize to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = torch.sign(normalized_action[..., -1])

    return normalized_action

def invert_gripper_action_tensor(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.clone()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_vla(cfg: Any) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    # Register OpenVLA model to HF Auto Classes to use LOCAL modeling code
    # This ensures we use the modified local code instead of remote cached code
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # If loading a locally stored pretrained checkpoint, update config and check model files
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    # Load the model using LOCAL registered class (not remote code)
    # [8BIT FIX] Add device_map="auto" when using quantization to prevent .to(device) errors
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "load_in_8bit": cfg.load_in_8bit,
        "load_in_4bit": cfg.load_in_4bit,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,  # Use local registered OpenVLAForActionPrediction class
    }
    if cfg.load_in_8bit or cfg.load_in_4bit:
        load_kwargs["device_map"] = "auto"
    
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        **load_kwargs
    )

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action normalization
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    """
    Apply FiLM (Feature-wise Linear Modulation) to the VLA vision backbone.

    Args:
        vla: The VLA model
        cfg: Configuration object with model parameters

    Returns:
        torch.nn.Module: VLA model with FiLM applied
    """
    from peft import LoraConfig, get_peft_model

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    # Create and apply FiLMed vision backbone
    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    # Load vision backbone checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    # Use the model component instead of wrapper and convert to bfloat16
    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)

    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    """
    Get proprioception projector for the VLA model.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model
        proprio_dim: Dimension of proprioception data

    Returns:
        ProprioProjector: The initialized proprio projector
    """
    # Initialize projector and move to device
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "proprio_projector--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "proprio_projector--300000_checkpoint.pt",
            "Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata": "proprio_projector--150000_checkpoint.pt",
            "openvla/openvla-7b-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
            "Haozhan72/openvla-oft-libero10-traj1-rl": "proprio_projector--150000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
    """
    Get noisy action projector for diffusion-based action prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        NoisyActionProjector: The initialized noisy action projector
    """
    # Initialize projector and move to device
    noisy_action_projector = NoisyActionProjector(
        llm_dim=llm_dim,
    ).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()

    # Find and load checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "noisy_action_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    noisy_action_projector.load_state_dict(state_dict)

    return noisy_action_projector


def get_action_head(cfg: Any, llm_dim: int) -> Union[L1RegressionActionHead, DiffusionActionHead]:
    """
    Get action head for continuous value prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        Union[L1RegressionActionHead, DiffusionActionHead]: The initialized action head

    Raises:
        AssertionError: If both L1 regression and diffusion are specified
    """
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot use both L1 regression and diffusion action head!"

    # Initialize appropriate action head based on configuration
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_diffusion_steps_train=cfg.num_diffusion_steps_train
        )
        # Set number of diffusion steps for inference
        action_head.noise_scheduler.set_timesteps(cfg.num_diffusion_steps_inference)
    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "action_head--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "action_head--300000_checkpoint.pt",
            "Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata": "action_head--150000_checkpoint.pt",
            "Haozhan72/openvla-oft-libero10-traj1-rl": "action_head--150000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_action_head_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

    return action_head


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images


def action_contrastive_fusion(selected_layer_action,final_layer_action,coffes):


    u = torch.from_numpy(final_layer_action).float()
    v = torch.from_numpy(selected_layer_action).float()

    # parallel decomp.
    dot_product = torch.dot(u,v)
    v_norm_sq = torch.dot(v,v)
    u_parallel = (dot_product / v_norm_sq) * v

    # parpendicular decomp.
    u_perp = u - u_parallel

    # apply fusion
    refined_vector = u + coffes * u_perp


    return refined_vector.numpy()






def extend_mask_after_last_true(mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mask: Bool tensor of shape [B, S] (True = valid, False = masked)
    Returns:
        new_mask: Bool tensor of shape [B, S] with all positions after
                  the last True set to True as well.
    """
    B, S = mask.shape
    device = mask.device

    last_true_idx = torch.where(mask, torch.arange(S, device=device).expand(B, S), -1)
    last_true_idx = last_true_idx.max(dim=1).values  # [B]

    arange = torch.arange(S, device=device).unsqueeze(0).expand(B, S)  # [B,S]
    new_mask = arange >= last_true_idx.unsqueeze(1)  # [B,S]

    new_mask = new_mask | mask
    return new_mask


def get_context_mask_for_inference(context_hidden, action_head, num_patches):
    """
    为 inference 构建 context mask，参考 retriving_data.py 中的实现
    
    Args:
        context_hidden: 最后一层的 hidden states (B, seq_len, D)
        action_head: action head 用于获取 action 维度信息
        num_patches: vision patches 的数量
        
    Returns:
        context_mask: 用于 energy model 的 context mask (B, seq_len)
    """
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM
    
    B, seq_len, D = context_hidden.shape
    device = context_hidden.device
    
    # 创建假的 labels 来获取 action masks
    # 假设 action tokens 在序列的后半部分
    fake_labels = torch.full((B, seq_len), -100, device=device)  # IGNORE_INDEX = -100
    
    # 设置 action 部分为有效 tokens
    action_start = num_patches + (seq_len - num_patches - 1) // 2  # 大概的 action 开始位置
    action_end = action_start + NUM_ACTIONS_CHUNK * ACTION_DIM
    fake_labels[:, action_start:action_end] = 1  # 设置为非 IGNORE_INDEX
    
    # 获取 action masks
    current_action_mask = get_current_action_mask(fake_labels)
    next_actions_mask = get_next_actions_mask(fake_labels)
    action_mask = current_action_mask | next_actions_mask
    action_mask = extend_mask_after_last_true(action_mask)
    
    # 创建 patch mask (vision patches 部分设为 False)
    patch_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)
    
    # 创建 eos mask (EOS token 部分设为 True)
    eos_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    
    # 拼接成完整的 context mask
    context_mask = torch.cat([patch_mask, action_mask, eos_mask], dim=1)
    
    return context_mask


def one_step_energy_correction_seq(energy_head, h, A_bc, energy_mask, alpha=0.1, clip_frac=0.2,
                                   act_range=None, correct_first_only=False):
    """
    A_bc: [H, Da] (numpy array or torch tensor)
    """
    if isinstance(A_bc, np.ndarray):
        A_bc = torch.tensor(A_bc, dtype=torch.bfloat16, device=h.device).unsqueeze(0)

    B, H, Da = A_bc.shape
    A = A_bc  # [B,H,Da]
    A = invert_gripper_action_tensor(normalize_gripper_action_tensor(A)).detach().clone().requires_grad_(True)
    A[..., -1] = torch.where(A[..., -1] == -1, 1, 0)


    with torch.enable_grad():
        E = energy_head(h, A, energy_mask)
        grad_A = torch.autograd.grad(E.sum(), A)[0]      # [B,H,Da]


    if correct_first_only:
        mask = torch.zeros_like(grad_A); mask[:,0,:] = 1.0
        grad_A = grad_A * mask

    # updates
    step = alpha * grad_A
    if act_range is not None:
        max_step = clip_frac * act_range.view(1,1,-1).to(step.device)
        step = torch.clamp(step, -max_step, max_step)
    else:
        step_norm = step.flatten(1).norm(dim=-1, keepdim=True) + 1e-6
        base_norm = A_bc.flatten(1).norm(dim=-1, keepdim=True) + 1e-6
        coef = torch.minimum(torch.ones_like(step_norm), (clip_frac*base_norm)/step_norm)
        step = step * coef.view(B,1,1)

    A_ref = A - step

    return A_ref.squeeze(0).detach().cpu().to(torch.float32).numpy()


def k_step_energy_correction_seq(
    energy_head,
    h,
    A_bc,
    energy_mask,
    k: int = 1,
    alpha: float = 0.1,
    clip_frac: float = 0.2,
    act_range=None,                # Optional[np.ndarray or torch.Tensor], per-dim range
    correct_first_only: bool = False,
):
    """
    k steps energy correction sequentially
    return :: [H, Da] 的 numpy float32。
    """
    device = h.device
    dtype  = torch.bfloat16
    

    # -- to torch [1,H,Da]
    if isinstance(A_bc, np.ndarray):
        A0 = torch.tensor(A_bc, dtype=dtype, device=device).unsqueeze(0)
    else:
        A0 = A_bc
        if A0.dim() == 2:
            A0 = A0.unsqueeze(0)
        A0 = A0.to(device=device, dtype=dtype)

    B, H_, Da = A0.shape
    assert B == 1, "该极简版假定 batch=1。"

    base_norm = A0.flatten(1).norm(dim=-1, keepdim=True) + 1e-6  # [1,1]


    A = invert_gripper_action_tensor(normalize_gripper_action_tensor(A0)).detach().clone()
    A[..., -1] = torch.where(A[..., -1] == -1, 1, 0)


    if act_range is not None:
        if isinstance(act_range, np.ndarray):
            act_range_t = torch.tensor(act_range, device=device, dtype=dtype)
        else:
            act_range_t = act_range.to(device=device, dtype=dtype)
        act_range_t = act_range_t.view(1, 1, -1)  # [1,1,Da]

    # k iterations of residule correction
    A_prev = A.clone()  # Keep previous valid A
    for iter_idx in range(max(1, int(k))):
        A = A.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            E = energy_head(h, A, energy_mask)
            E_sum = E.sum() if E.dim() > 0 else E
            grad_A = torch.autograd.grad(E_sum, A, create_graph=False, retain_graph=False)[0]  # [1,H,Da]

        # Check for NaN/Inf in gradient
        if not torch.isfinite(grad_A).all():
            print(f"Warning: NaN/Inf detected in gradient at iteration {iter_idx}, E={E.item():.6f}, skipping")
            A = A_prev.clone()
            break

        # Check gradient magnitude
        grad_norm = grad_A.flatten(1).norm(dim=-1).item()
        if grad_norm < 1e-8:
            print(f"Warning: Gradient too small (norm={grad_norm:.2e}) at iteration {iter_idx}, E={E.item():.6f}, stopping")
            break

        if correct_first_only:
            mask = torch.zeros_like(grad_A); mask[:, 0, :] = 1.0
            grad_A = grad_A * mask

        step = alpha * grad_A

        if act_range is not None:
            max_step = (clip_frac * act_range_t).to(step.dtype)
            step = torch.clamp(step, -max_step, max_step)
        else:
            step_norm = step.flatten(1).norm(dim=-1, keepdim=True) + 1e-6
            coef = torch.minimum(torch.ones_like(step_norm), (clip_frac * base_norm) / step_norm)
            step = step * coef.view(1, 1, 1)

        A_new = (A - step).detach()
        
        # Check for NaN in updated A
        if torch.isfinite(A_new).all():
            # Check if energy actually decreased (optional check, can be expensive)
            # For now, just update if finite
            A = A_new
            A_prev = A.clone()
        else:
            print(f"Warning: NaN detected in updated action at iteration {iter_idx}, using previous valid action")
            A = A_prev.clone()
            break

    energy_head.eval()
    A_corrected = A.detach().clone().requires_grad_(True)
    
    # Check and fix NaN values in A_corrected before processing
    if not torch.isfinite(A_corrected).all():
        print(f"Warning: NaN detected in A_corrected, replacing with A0")
        # Use A0 as fallback, but need to process it the same way as A was processed
        A_corrected = A0.clone()
        A_corrected = normalize_gripper_action_tensor(A_corrected)
        # Invert gripper action (multiply last dimension by -1)
        A_corrected[..., -1] *= -1.0
        A_corrected = A_corrected.detach().clone()
        A_corrected[..., -1] = torch.where(A_corrected[..., -1] == -1, 1, 0)
        A_corrected = A_corrected.requires_grad_(True)
    
    A_corrected[..., -1] = torch.round(A_corrected[..., -1]).clamp(0, 1)
    E_corrected = energy_head(h, A_corrected, energy_mask)
    energy_head.train()

    print(f"Action Energy: {E.item():.10f} | Corrected Action Energy: {E_corrected.item():.10f}")
    return A.squeeze(0).detach().cpu().to(torch.float32).numpy()

def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    h_head = None,
    return_eos_info: bool = False,
    eos_head: Optional[torch.nn.Module] = None,
    eos_threshold: float = 0.5,
    auto_regression: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], bool, Optional[int]]]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM
        h_head: Optional energy model head
        return_eos_info: Whether to return EOS detection info
        eos_head: Optional EOS classification head for substep boundary detection
        eos_threshold: Threshold for EOS detection (default: 0.5)
        auto_regression: Only used when action_head is None (discrete token mode).
                         True  → vla.generate() autoregressive (56 forward passes).
                         False → predict_action parallel decode (1 forward pass).

    Returns:
        List[np.ndarray]: Predicted actions, or tuple (actions, has_eos, eos_position) if return_eos_info=True
    """
    global _FLOPS_CALCULATED
    
    with torch.inference_mode():

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)

    
        # Build VLA prompt
        if cfg.remove_wrap:
            prompt = task_label.lower()
        else:
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process primary image
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)


        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            # Concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        # Start timing for VLA forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vla_start_time = time.time()

        # Initialize EOS detection variables
        has_eos = False
        eos_position = None

        # Generate action
        if action_head is None:
            if auto_regression:
                # 自回归路径：vla.generate() 逐 token 生成，共 56 次 forward pass
                vanilla_input_ids = inputs["input_ids"]
                if not torch.all(vanilla_input_ids[:, -1] == 29871):
                    vanilla_input_ids = torch.cat(
                        (vanilla_input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(vanilla_input_ids.device)),
                        dim=1,
                    )
                vanilla_inputs = {**inputs, "input_ids": vanilla_input_ids}
                action_dim = vla.get_action_dim(cfg.unnorm_key)
                generated_ids = vla.generate(**vanilla_inputs, max_new_tokens=action_dim, do_sample=False)
                predicted_action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
                discretized_actions = vla.vocab_size - predicted_action_token_ids
                discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
                normalized_actions = vla.bin_centers[discretized_actions]
                action = vla._unnormalize_actions(normalized_actions, cfg.unnorm_key)
                hiddens, layer_actions, energy_pad_mask = None, [], None
            else:
                # 并行 decode：追加 56 个 placeholder token，1 次 forward pass 同时得到所有位置的
                # categorical distribution，argmax → token IDs → bin_centers → unnorm。
                # 走 _regression_or_discrete_prediction 中 action_head=None 的 discrete 分支。
                result = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    action_head=None,
                    use_film=use_film,
                )
                action, hiddens, layer_actions, energy_pad_mask = result
            has_eos, eos_position = False, None
        else:
            # Custom action head for continuous actions
            if cfg.h_decoding:
                if return_eos_info:
                    result = vla.predict_action(
                        **inputs,
                        unnorm_key=cfg.unnorm_key,
                        do_sample=False,
                        proprio=proprio,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        action_head=action_head,
                        use_film=use_film,
                        return_eos_info=True,
                        eos_head=eos_head,
                        eos_threshold=eos_threshold,
                    )
                    if len(result) == 6:
                        action, hiddens, layer_actions, _, has_eos, eos_position = result
                    else:
                        action, hiddens, layer_actions = result[:3]
                        has_eos, eos_position = False, None
                else:
                    action, hiddens, layer_actions = vla.predict_action(
                        **inputs,
                        unnorm_key=cfg.unnorm_key,
                        do_sample=False,
                        proprio=proprio,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        action_head=action_head,
                        use_film=use_film,
                    )
                    has_eos, eos_position = False, None
            else:
                try:
                    if return_eos_info:
                        result = vla.predict_action(  # in case of our implementation
                            **inputs,
                            unnorm_key=cfg.unnorm_key,
                            do_sample=False,
                            proprio=proprio,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            action_head=action_head,
                            use_film=use_film,
                            return_eos_info=True,
                            eos_head=eos_head,
                            eos_threshold=eos_threshold,
                        )
                        if len(result) == 6:
                            action, hiddens, layer_actions, energy_pad_mask, has_eos, eos_position = result
                        else:
                            action, hiddens, layer_actions, energy_pad_mask = result[:4]
                            has_eos, eos_position = False, None
                    else:
                        action, hiddens, layer_actions, energy_pad_mask = vla.predict_action(  # in case of our implementation
                            **inputs,
                            unnorm_key=cfg.unnorm_key,
                            do_sample=False,
                            proprio=proprio,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            action_head=action_head,
                            use_film=use_film,
                        )
                        has_eos, eos_position = False, None
                except ValueError as e:
                    if return_eos_info:
                        result = vla.predict_action(  # in case of baseline
                            **inputs,
                            unnorm_key=cfg.unnorm_key,
                            do_sample=False,
                            proprio=proprio,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            action_head=action_head,
                            use_film=use_film,
                            return_eos_info=True,
                            eos_head=eos_head,
                            eos_threshold=eos_threshold,
                        )
                        if len(result) == 6:
                            action, hiddens, _, _, has_eos, eos_position = result
                        else:
                            action, hiddens = result[:2]
                            has_eos, eos_position = False, None
                    else:
                        action, hiddens = vla.predict_action(  # in case of baseline
                            **inputs,
                            unnorm_key=cfg.unnorm_key,
                            do_sample=False,
                            proprio=proprio,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            action_head=action_head,
                            use_film=use_film,
                        )
                        has_eos, eos_position = False, None

        # End timing for VLA forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vla_end_time = time.time()
        vla_elapsed_time = (vla_end_time - vla_start_time) * 1000  # Convert to ms
        
        # Print VLA forward pass timing
        # print(f"\n{'='*80}")
        # print(f"[TIMING] VLA Forward Pass: {vla_elapsed_time:.2f} ms ({vla_elapsed_time/1000:.4f} s)")
        # print(f"{'='*80}\n")
        
        # Calculate FLOPs only once
        if not _FLOPS_CALCULATED:
            try:
                from thop import profile, clever_format
                
                print(f"\n{'='*80}")
                print(f"[MODEL INFO]")
                print(f"  Model type: {type(vla).__name__}")
                print(f"  Input shape: pixel_values={inputs['pixel_values'].shape}")
                if 'input_ids' in inputs:
                    print(f"  Input shape: input_ids={inputs['input_ids'].shape}")
                
                # Try to profile the model's forward pass
                # Note: This may not capture the full pipeline including action_head
                try:
                    # Attempt to profile using the vision backbone and language model separately
                    # since full model profiling with custom methods is complex
                    
                    # Method 1: Try to get basic parameter count
                    total_params = sum(p.numel() for p in vla.parameters())
                    params_readable = clever_format([total_params], "%.3f")[0]
                    
                    print(f"\n[MODEL STATISTICS]")
                    print(f"  Total Parameters: {params_readable} ({total_params:,})")
                    
                    # Method 2: Estimate FLOPs based on common VLA architecture
                    # For transformer models: FLOPs ≈ 2 * params * sequence_length
                    # This is a rough approximation
                    seq_len = inputs['input_ids'].shape[1]
                    if hasattr(vla, 'vision_backbone'):
                        # Add vision processing FLOPs estimation
                        vision_flops = 0
                        if hasattr(vla.vision_backbone, 'parameters'):
                            vision_params = sum(p.numel() for p in vla.vision_backbone.parameters())
                            # Vision processing: params * image_patches
                            num_patches = (224 // 16) ** 2  # Assuming 16x16 patches for 224x224 image
                            vision_flops = 2 * vision_params * num_patches
                    
                    # Estimate total FLOPs (very rough approximation)
                    estimated_flops = 2 * total_params * seq_len
                    flops_readable = clever_format([estimated_flops], "%.3f")[0]
                    
                    print(f"\n[FLOPS ESTIMATION] (Approximate)")
                    print(f"  Estimated FLOPs per inference: {flops_readable} ({estimated_flops:.2e})")
                    print(f"  Note: This is a rough estimate for transformer forward pass")
                    
                    # Estimate FLOPs per second
                    # if vla_elapsed_time > 0:
                    #     flops_per_sec = estimated_flops / (vla_elapsed_time / 1000)
                    #     gflops = flops_per_sec / 1e9
                    #     tflops = flops_per_sec / 1e12
                    #     print(f"  Estimated GFLOPS: {gflops:.2f} GFLOP/s")
                    #     print(f"  Estimated TFLOPS: {tflops:.4f} TFLOP/s")
                    
                except Exception as profile_err:
                    print(f"\n[INFO] Could not compute FLOPs statistics: {profile_err}")
                    print(f"[INFO] Transformer models with custom predict_action may require manual profiling.")
                
                print(f"{'='*80}\n")
                _FLOPS_CALCULATED = True
                
            except ImportError:
                print("\n[WARNING] thop library not available. Skipping FLOPs calculation.")
                print("[INFO] Install with: pip install thop\n")
                _FLOPS_CALCULATED = True
            except Exception as e:
                print(f"\n[WARNING] Could not calculate FLOPs: {e}\n")
                _FLOPS_CALCULATED = True

        if cfg.h_decoding:

            residule_coef = 0.1
            f_list = compute_hamitonians(layer_actions,proprio,h_head,DEVICE)
            selected_index = select_layer_index(f_list)

            # prevs
            # contrast = layer_actions[-1] - layer_actions[selected_index]
            # action = action + residule_coef * contrast

            # hiddent contrast
            # contrast = hiddens[-1] - hiddens[selected_index]
            # final_hidden = hiddens[-1] + residule_coef * contrast
            # action = action_head.predict_action(final_hidden)

            # ACF on actions on space and orientation
            candidate_actions = layer_actions[selected_index] # (8,7)
            model_actions = layer_actions[-1]

            candidate_pos = candidate_actions[:,:3] # (8,3)
            model_pos = model_actions[:,:3] # (8,3)
            candidate_rot = candidate_actions[:,3:6] # (8,3)
            model_rot = model_actions[:,3:6] # (8,3)

            num_action_chunk = candidate_pos.shape[0]

            for i in range(num_action_chunk):
                model_pos[i] = action_contrastive_fusion(candidate_pos[i],model_pos[i],residule_coef)
                model_rot[i] = action_contrastive_fusion(candidate_rot[i],model_rot[i],residule_coef)
            
            model_actions[:,:3] = model_pos
            model_actions[:,3:6] = model_rot

            action = model_actions
        
    if cfg.e_decoding:
        # action = one_step_energy_correction_seq(h_head,hiddens[-1],action,energy_pad_mask)
        action = k_step_energy_correction_seq(h_head,hiddens[-1],action,energy_pad_mask,cfg.energy_k,alpha=cfg.energy_alpha)




    # Return action chunk as list of actions
    # Vanilla OpenVLA returns a single action of shape [7]; OFT returns [chunk_size, 7]
    if isinstance(action, np.ndarray) and action.ndim == 1:
        actions_list = [action]
    else:
        actions_list = [action[i] for i in range(len(action))]
    
    if return_eos_info:
        # Return actions with EOS detection info
        # has_eos and eos_position are set during predict_action call above
        return actions_list, has_eos, eos_position
    else:
        return actions_list


def quat2axisangle_torch(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    quat2axisangle： (B,4) -> (B,3)
    quat[..., :3] = (x,y,z), quat[..., 3] = w
    """
 
    xyz = quat[..., :3]          # (B,3)
    w   = quat[..., 3].clamp(-1.0, 1.0)  # (B,)


    angle = 2.0 * torch.acos(w)  # (B,)

    denom = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))  # (B,)
    safe_denom = denom.clone().masked_fill_(denom < eps, 1.0)

    axis_angle = xyz * (angle / safe_denom).unsqueeze(-1)  # (B,3)

    axis_angle = axis_angle.masked_fill(denom.unsqueeze(-1) < eps, 0.0)
    return axis_angle

def compute_hamitonians(layer_actions, props, h_head,DEVICE):

    props = torch.from_numpy(props).to(DEVICE, dtype=torch.bfloat16)

    # compute_raw_coordinates
    ori_pos   = props[:3]               # (3)  
    ori_quat  = props[3:7]              # (4)
    ori_rot   = quat2axisangle_torch(ori_quat)  # (3)
    ori_coords= torch.cat([ori_pos, ori_rot], dim=-1)  # (6)
    f1f2_list = []

    for i in range(len(layer_actions)):

        # raw action coordinates
        pred_actions = layer_actions[i][:, :6]
        pred_actions = torch.from_numpy(pred_actions).to(DEVICE, dtype=torch.bfloat16) # to tensor


        abs_pred = torch.cumsum(pred_actions, dim=0) + ori_coords.unsqueeze(0)   # (T, D)

        # position and velocities
        z  = abs_pred[:-2, :]  # (T-2, D)
        z_next  = abs_pred[1:-1,    :]
        dz_dt   = z_next   - z   # (T-2, D)
        z_qp  = torch.cat([z, dz_dt],   dim=-1)  # (T-2, 2D)
        
        F1_F2   = h_head(z_qp)      # (T-2, 2)
        f1f2_list.append(F1_F2) # list of (T-2,2)
    
    return f1f2_list

def select_layer_index(f_list):

    # select layer with maximum H difference and returns the layer index

    target = f_list[-1]
    mses = []
    for idx, t in enumerate(f_list[:-1]):
        # mse = mean((t - target)^2)
        mse = F.mse_loss(t, target, reduction='mean')
        mses.append(mse)

    mses_tensor = torch.stack(mses)
    min_idx = torch.argmin(mses_tensor).item()
    return min_idx

    

    


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()