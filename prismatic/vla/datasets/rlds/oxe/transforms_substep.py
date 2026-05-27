"""
transforms_substep.py

Extended transforms for RLDS datasets that preserve episode-level metadata.
This module adds episode tracking capabilities to support per-timestep substep instructions.
"""

from typing import Any, Dict

import tensorflow as tf

from prismatic.vla.datasets.rlds.oxe.transforms import libero_dataset_transform


# Global episode counter for tracking episodes across dataset iterations
# Note: This is a stateful approach that works for single-dataset training
# For multi-dataset scenarios, consider using trajectory metadata
_episode_counter = None


def reset_episode_counter():
    """Reset the global episode counter. Call this when starting a new dataset iteration."""
    global _episode_counter
    _episode_counter = 0


def get_and_increment_episode_id():
    """Get current episode ID and increment counter."""
    global _episode_counter
    if _episode_counter is None:
        _episode_counter = 0
    current = _episode_counter
    _episode_counter += 1
    return current


def libero_dataset_transform_with_episode_id(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extended LIBERO dataset transform that adds episode ID tracking.
    
    This transform wraps the original libero_dataset_transform and adds an episode_id field
    to each timestep in the trajectory. The episode_id is a sequential integer that identifies
    which episode this trajectory belongs to.
    
    Args:
        trajectory: Dictionary of batched features (has leading time dimension)
        
    Returns:
        Dictionary with additional 'episode_id' field containing the episode index
        
    Note:
        - Episode IDs are sequential integers starting from 0
        - The counter is global and stateful across the dataset
        - Call reset_episode_counter() when starting a new training run
    """
    # Apply original LIBERO transform first
    trajectory = libero_dataset_transform(trajectory)
    
    # Get episode ID for this trajectory (Python int)
    episode_id = get_and_increment_episode_id()
    
    # Get trajectory length
    traj_len = tf.shape(trajectory["action"])[0]
    
    # Add episode_id as a repeated field for all timesteps in this trajectory
    # Convert Python int to TensorFlow constant first, then repeat
    # This will be propagated through the pipeline to each individual frame
    episode_id_tensor = tf.constant(episode_id, dtype=tf.int32)
    trajectory["episode_id"] = tf.repeat(episode_id_tensor, traj_len)
    
    return trajectory


def libero_dataset_transform_with_file_path(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Alternative LIBERO transform that extracts episode ID from file_path metadata.
    
    This approach is more robust than global counters but requires that the file_path
    contains episode information. It attempts to extract episode number from patterns like:
    - "task_name_demo_5.hdf5" -> episode_id = 5
    - "episode_10/demo.hdf5" -> episode_id = 10
    
    Args:
        trajectory: Dictionary of batched features with traj_metadata
        
    Returns:
        Dictionary with episode_id field
        
    Note:
        Falls back to 0 if episode ID cannot be extracted from file_path
    """
    # Apply original LIBERO transform
    trajectory = libero_dataset_transform(trajectory)
    
    # Try to extract episode ID from trajectory metadata
    episode_id = 0  # Default fallback
    
    if "traj_metadata" in trajectory:
        metadata = trajectory["traj_metadata"]
        if "episode_metadata" in metadata:
            ep_metadata = metadata["episode_metadata"]
            
            # Try to extract from file_path if available
            if "file_path" in ep_metadata:
                try:
                    file_path = ep_metadata["file_path"]
                    # Parse file_path to extract episode number
                    # This is dataset-specific and may need adjustment
                    # For now, use a simple pattern matching approach
                    import re
                    
                    # Try patterns like "demo_5", "episode_10", etc.
                    patterns = [
                        r"demo_(\d+)",
                        r"episode_(\d+)",
                        r"ep_(\d+)",
                        r"traj_(\d+)",
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, str(file_path))
                        if match:
                            episode_id = int(match.group(1))
                            break
                except Exception:
                    # If extraction fails, use default
                    pass
    
    # Get trajectory length
    traj_len = tf.shape(trajectory["action"])[0]
    
    # Add episode_id field
    # Convert Python int to TensorFlow constant first
    episode_id_tensor = tf.constant(episode_id, dtype=tf.int32)
    trajectory["episode_id"] = tf.repeat(episode_id_tensor, traj_len)
    
    return trajectory


# Registry for substep-aware transforms
OXE_STANDARDIZATION_TRANSFORMS_WITH_EPISODE_ID = {
    # LIBERO datasets with episode tracking
    "libero_spatial_no_noops": libero_dataset_transform_with_episode_id,
    "libero_object_no_noops": libero_dataset_transform_with_episode_id,
    "libero_goal_no_noops": libero_dataset_transform_with_episode_id,
    "libero_10_no_noops": libero_dataset_transform_with_episode_id,
    "libero_4_task_suites_no_noops": libero_dataset_transform_with_episode_id,
}

