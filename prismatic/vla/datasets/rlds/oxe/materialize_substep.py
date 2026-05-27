"""
materialize_substep.py

Extended factory functions for initializing Open-X Embodiment datasets with episode ID tracking.
This module provides wrappers around the original materialize.py functions but uses transforms
that preserve episode-level metadata for substep instruction support.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
from prismatic.vla.datasets.rlds.oxe.transforms_substep import (
    OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID,
    reset_episode_counter,
)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def make_oxe_dataset_kwargs_with_episode_id(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Dict[str, Any]:
    """
    Generates config (kwargs) for given dataset with episode ID tracking enabled.
    
    This is an extended version of make_oxe_dataset_kwargs that uses transforms
    from transforms_substep.py to preserve episode IDs throughout the data pipeline.
    
    Args:
        dataset_name: Name of the dataset (e.g., "libero_goal_no_noops")
        data_root_dir: Root directory containing RLDS datasets
        load_camera_views: Camera views to load (e.g., ("primary", "wrist"))
        load_depth: Whether to load depth images
        load_proprio: Whether to load proprioceptive state
        load_language: Whether to load language instructions
        action_proprio_normalization_type: Type of normalization for actions/proprio
        
    Returns:
        Dictionary of dataset kwargs that can be passed to make_dataset_from_rlds_with_episode_id
        
    Note:
        Currently only supports LIBERO datasets. Other datasets will fall back to
        standard transforms without episode tracking.
    """
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    
    # Validate action encoding
    if dataset_kwargs["action_encoding"] not in [
        ActionEncoding.EEF_POS,
        ActionEncoding.EEF_R6,
        ActionEncoding.JOINT_POS_BIMANUAL,
    ]:
        raise ValueError(
            f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 & JOINT_POS_BIMANUAL actions supported!"
        )

    # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL:
        dataset_kwargs["absolute_action_mask"] = [True] * 14
        dataset_kwargs["action_normalization_mask"] = [True] * 14
    
    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Adjust Loaded Camera Views
    if len(missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"]))) > 0:
        raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter camera views
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop("state_encoding")
    dataset_kwargs.pop("action_encoding")
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # [CRITICAL] Use episode-tracking transforms instead of standard transforms
    if dataset_name in OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID:
        dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID[dataset_name]
        overwatch.info(f"Using episode-tracking transform for dataset: {dataset_name}")
    else:
        # Fall back to standard transforms for datasets without episode tracking
        from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
        dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]
        overwatch.warning(
            f"Dataset {dataset_name} does not have episode-tracking transform. "
            f"Episode IDs will default to 0."
        )

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights_with_episode_id(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs with episode tracking for a dataset mix from Open X-Embodiment.
    
    Extended version of get_oxe_dataset_kwargs_and_weights that uses episode-tracking transforms.
    
    Args:
        data_root_dir: Base directory containing RLDS/TFDS-formatted datasets
        mixture_spec: List of (dataset_name, sampling_weight) tuples
        load_camera_views: Camera views to load
        load_depth: Load depth information
        load_proprio: Load proprioceptive state
        load_language: Load language instructions
        action_proprio_normalization_type: Type of normalization
        
    Returns:
        Tuple of (per_dataset_kwargs_list, sampling_weights)
        
    Note:
        - Resets the global episode counter before creating dataset kwargs
        - Only datasets with episode-tracking transforms will preserve episode IDs
    """
    # Reset episode counter at the start of dataset creation
    reset_episode_counter()
    overwatch.info("Reset episode counter for new dataset initialization")
    
    per_dataset_kwargs = []
    weights = []
    
    for dataset_name, weight in mixture_spec:
        dataset_kwargs = make_oxe_dataset_kwargs_with_episode_id(
            dataset_name,
            data_root_dir,
            load_camera_views=load_camera_views,
            load_depth=load_depth,
            load_proprio=load_proprio,
            load_language=load_language,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )
        per_dataset_kwargs.append(dataset_kwargs)
        weights.append(weight)

    return per_dataset_kwargs, weights

