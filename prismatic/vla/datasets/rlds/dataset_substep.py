"""
dataset_substep.py

Extended RLDS dataset loader that preserves episode-level metadata for substep instruction support.
Based on the original dataset.py but modified to track episode IDs through the data pipeline.

Key difference from original:
- Uses standardize_fn with episode tracking during data loading
- Uses original transform (without episode tracking) for statistics computation
"""

import copy
import inspect
import json
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.utils.data_utils import get_dataset_statistics, normalize_action_and_proprio

# Disable GPU for TensorFlow (prevents conflicts with PyTorch)
tf.config.set_visible_devices([], "GPU")


def make_dataset_from_rlds_with_episode_id(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: ACTION_PROPRIO_NORMALIZATION_TYPE,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """
    Extended version of make_dataset_from_rlds that preserves episode IDs.
    
    This function loads RLDS datasets and adds episode_id tracking to each frame.
    The episode_id field is preserved through the data pipeline and can be used
    to query per-timestep substep instructions.
    
    Args:
        Same as make_dataset_from_rlds, see original function for full documentation.
        
    Returns:
        Tuple of (dataset, dataset_statistics) where dataset includes episode_id field
        
    Additional Fields in Output:
        - observation/timestep: timestep index within episode
        - episode_id: episode index (added by standardize_fn, preserved here)
        
    Note:
        The standardize_fn must add an 'episode_id' field to the trajectory.
        Use transforms from transforms_substep.py for LIBERO datasets.
    """
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        """
        Restructure trajectory into standardized format, preserving episode_id.
        
        This function is similar to the original but preserves the episode_id field
        added by the standardize_fn (e.g., libero_dataset_transform_with_episode_id).
        """
        # Apply standardization function if provided
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
                "Did you write a `standardize_fn`?"
            )

        # Extract images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    (
                        tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                        if key is None
                        else tf.cast(old_obs[key], tf.float32)
                    )
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # Add timestep info
        new_obs["timestep"] = tf.range(traj_len)
        
        # [CRITICAL] Add episode_id to observation dict so it gets processed correctly by chunk_act_obs
        # This field is added by standardize_fn (e.g., libero_dataset_transform_with_episode_id)
        if "episode_id" in traj:
            # Ensure episode_id is int32 tensor, not Python int
            new_obs["episode_id"] = tf.cast(traj["episode_id"], tf.int32)
        else:
            # If episode_id is not provided, default to 0
            # This maintains compatibility with datasets that don't use episode tracking
            new_obs["episode_id"] = tf.repeat(0, traj_len)

        # Extract language_key into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, "
                    "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)

        # Build output trajectory dictionary
        traj_output = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj_output["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj_output

    builder = tfds.builder(name, data_dir=data_dir)

    # Load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
        
        # Convert lists to numpy arrays for TensorFlow compatibility
        # JSON serialization converts numpy arrays to lists, need to convert back
        for key in ["action", "proprio"]:
            if key in dataset_statistics:
                for stat_key in ["mean", "std", "min", "max", "q01", "q99"]:
                    if stat_key in dataset_statistics[key]:
                        dataset_statistics[key][stat_key] = np.array(dataset_statistics[key][stat_key])
    elif dataset_statistics is None:
        # [CRITICAL] For statistics computation, use original transform WITHOUT episode tracking
        # This prevents episode_id field from interfering with statistics calculation
        from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
        
        # Create a temporary standardize function without episode tracking
        standardize_fn_for_stats = OXE_STANDARDIZATION_TRANSFORMS.get(name, standardize_fn)
        
        # Create a temporary restructure function for statistics
        def restructure_for_stats(traj):
            """Restructure without episode tracking for statistics computation."""
            if standardize_fn_for_stats is not None:
                traj = standardize_fn_for_stats(traj)
            
            if not all(k in traj for k in REQUIRED_KEYS):
                raise ValueError(
                    f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
                    "Did you write a `standardize_fn`?"
                )
            
            # Extract images, depth images and proprio from the "observation" dict
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}
            
            for new, old in image_obs_keys.items():
                if old is None:
                    new_obs[f"image_{new}"] = tf.repeat("", traj_len)
                else:
                    new_obs[f"image_{new}"] = old_obs[old]
            
            for new, old in depth_obs_keys.items():
                if old is None:
                    new_obs[f"depth_{new}"] = tf.repeat("", traj_len)
                else:
                    new_obs[f"depth_{new}"] = old_obs[old]
            
            if state_obs_keys:
                new_obs["proprio"] = tf.concat(
                    [
                        (
                            tf.zeros((traj_len, 1), dtype=tf.float32)
                            if key is None
                            else tf.cast(old_obs[key], tf.float32)
                        )
                        for key in state_obs_keys
                    ],
                    axis=1,
                )
            
            new_obs["timestep"] = tf.range(traj_len)
            
            task = {}
            if language_key is not None:
                if traj[language_key].dtype != tf.string:
                    raise ValueError(f"Language key {language_key} has dtype {traj[language_key].dtype}, but it must be tf.string.")
                task["language_instruction"] = traj.pop(language_key)
            
            # No episode_id here - this is for statistics only
            traj_output = {
                "observation": new_obs,
                "task": task,
                "action": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(name, traj_len),
            }
            
            if absolute_action_mask is not None:
                if len(absolute_action_mask) != traj["action"].shape[-1]:
                    raise ValueError(
                        f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                        f"does not match action dimension ({traj['action'].shape[-1]})."
                    )
                traj_output["absolute_action_mask"] = tf.tile(
                    tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                    [traj_len, 1],
                )
            
            return traj_output
        
        # Compute statistics using transform WITHOUT episode tracking
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure_for_stats, num_parallel_calls)
        
        # Try to load from cache, otherwise compute on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn_for_stats) if standardize_fn_for_stats is not None else "",
            ),
            save_dir=data_dir,
        )

    # [CRITICAL] Ensure all statistics are numpy arrays, not lists
    # This is necessary for TensorFlow operations in normalize_action_and_proprio
    # Convert any list-type statistics to numpy arrays
    for key in ["action", "proprio"]:
        if key in dataset_statistics:
            for stat_key in ["mean", "std", "min", "max", "q01", "q99"]:
                if stat_key in dataset_statistics[key]:
                    if isinstance(dataset_statistics[key][stat_key], (list, tuple)):
                        dataset_statistics[key][stat_key] = np.array(dataset_statistics[key][stat_key])
    
    # [Important] Add action_normalization_mask to dataset_statistics if provided
    # This prevents normalization of specific action dimensions (e.g., gripper)
    if action_normalization_mask is not None:
        # Get action dimension - handle both numpy arrays and lists
        action_mean = dataset_statistics["action"]["mean"]
        if isinstance(action_mean, (list, tuple)):
            action_dim = len(action_mean)
        else:
            action_dim = action_mean.shape[-1]
        
        if len(action_normalization_mask) != action_dim:
            raise ValueError(
                f"Length of action_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({action_dim})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # Create dataset
    dataset = dl.DLataset.from_rlds(builder, split="train" if train else "val", shuffle=shuffle, num_parallel_reads=num_parallel_reads)
    dataset = dataset.traj_map(restructure, num_parallel_calls=num_parallel_calls)
    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics

