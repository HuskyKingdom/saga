"""
run_expert_replay.py

Replays a single expert demonstration episode from LIBERO RLDS dataset,
annotates each frame with timestep info, action type, and APD plan step,
then saves as a video.

Usage:
    python run_expert_replay.py \
        --rlds_data_dir /path/to/modified_libero_rlds \
        --substep_labels_path substep_labels_output.json \
        --suite libero_spatial_no_noops \
        --episode_idx 0 \
        --output_dir ./expert_replay_videos
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from libero.libero import benchmark

# Disable GPU for data loading
tf.config.set_visible_devices([], 'GPU')

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_substep_labels(substep_labels_path: str) -> Dict:
    """
    Load substep labels from JSON file.
    
    Args:
        substep_labels_path: Path to substep_labels_output.json
    
    Returns:
        Dictionary containing substep labels
    """
    logger.info(f"Loading substep labels from {substep_labels_path}")
    
    with open(substep_labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    return labels_data


def load_rlds_episode(rlds_data_dir: str, suite_name: str, episode_idx: int) -> Optional[Dict]:
    """
    Load a specific episode from RLDS dataset.
    
    Args:
        rlds_data_dir: RLDS dataset root directory
        suite_name: Task suite name (e.g., "libero_spatial_no_noops")
        episode_idx: Episode index to load
    
    Returns:
        Episode dictionary containing steps, or None if not found
    """
    dataset_path = os.path.join(rlds_data_dir, suite_name) + "/1.0.0/"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return None
    
    try:
        # Load using tensorflow_datasets builder
        builder = tfds.builder_from_directory(dataset_path)
        dataset = builder.as_dataset(split='train')
        
        # Get specific episode
        for idx, episode in enumerate(dataset):
            if idx == episode_idx:
                logger.info(f"Loaded episode {episode_idx} from {suite_name}")
                return episode
        
        logger.error(f"Episode {episode_idx} not found in dataset")
        return None
    
    except Exception as e:
        logger.error(f"Failed to load episode: {e}")
        return None


def extract_episode_data(episode: Dict) -> Tuple[np.ndarray, str, str]:
    """
    Extract actions and instruction from RLDS episode.
    
    Args:
        episode: RLDS episode dictionary
    
    Returns:
        (actions, language_instruction, task_name)
        - actions: (T, 7) action array
        - language_instruction: instruction string (with scene prefix removed)
        - task_name: task name extracted from file_path
    """
    steps = episode['steps']
    
    # Extract actions
    actions = []
    language_instruction = ""
    task_name = ""
    
    for step_idx, step in enumerate(steps):
        # Action
        action = step['action'].numpy()
        actions.append(action)
    
    # Get language instruction from episode metadata
    if 'episode_metadata' in episode:
        metadata = episode['episode_metadata']
        
        # Try to extract from file_path
        if 'file_path' in metadata:
            try:
                file_path = metadata['file_path'].numpy().decode('utf-8')
                filename = os.path.basename(file_path)
                if filename.endswith('_demo.hdf5'):
                    task_name = filename[:-10]  # Remove '_demo.hdf5'
                    language_instruction = task_name.replace('_', ' ')
            except Exception as e:
                logger.debug(f"Failed to extract from file_path: {e}")
        
        # Try standard keys if file_path extraction didn't work
        if not language_instruction:
            for key in ['language_instruction', 'instruction', 'task']:
                if key in metadata:
                    try:
                        language_instruction = metadata[key].numpy().decode('utf-8')
                        task_name = language_instruction.replace(' ', '_')
                        break
                    except:
                        pass
    
    # Clean up instruction: remove scene prefixes like "living room scene5", "kitchen scene3", etc.
    language_instruction = language_instruction.lower().strip()
    
    # Remove common scene prefixes using regex
    # Pattern matches: "living room scene5 ", "LIVING_ROOM_SCENE5_", etc.
    # For space-separated format
    scene_pattern_spaces = r'^(living\s+room|kitchen|study|bedroom|bathroom|office)\s+scene\d+\s+'
    language_instruction = re.sub(scene_pattern_spaces, '', language_instruction, flags=re.IGNORECASE)
    
    # For underscore-separated format (in task_name)
    scene_pattern_underscores = r'^(LIVING_ROOM|KITCHEN|STUDY|BEDROOM|BATHROOM|OFFICE)_SCENE\d+_'
    task_name = re.sub(scene_pattern_underscores, '', task_name, flags=re.IGNORECASE)
    
    return np.array(actions), language_instruction, task_name


def find_episode_labels(substep_labels: Dict, suite_name: str, task_name: str, episode_idx: int) -> Optional[Dict]:
    """
    Find substep labels for specific episode.
    
    Args:
        substep_labels: Full substep labels dictionary
        suite_name: Suite name
        task_name: Task name
        episode_idx: Episode index
    
    Returns:
        Episode labels dictionary or None if not found
    """
    # Remove "_no_noops" suffix from suite name for lookup
    suite_short = suite_name.replace('_no_noops', '')
    
    if suite_short not in substep_labels:
        logger.warning(f"Suite '{suite_short}' not found in substep labels")
        return None
    
    # Try exact match first
    if task_name in substep_labels[suite_short]:
        episode_key = f"episode_{episode_idx}"
        if episode_key in substep_labels[suite_short][task_name]:
            return substep_labels[suite_short][task_name][episode_key]
    
    # If exact match fails, try fuzzy matching (in case of scene prefix differences)
    task_name_lower = task_name.lower()
    for label_task_name in substep_labels[suite_short].keys():
        label_task_name_lower = label_task_name.lower()
        # Check if one is contained in the other (handles scene prefix differences)
        if task_name_lower in label_task_name_lower or label_task_name_lower in task_name_lower:
            logger.info(f"Fuzzy matched task '{task_name}' to '{label_task_name}'")
            episode_key = f"episode_{episode_idx}"
            if episode_key in substep_labels[suite_short][label_task_name]:
                return substep_labels[suite_short][label_task_name][episode_key]
    
    logger.warning(f"Task '{task_name}' not found in substep labels for suite '{suite_short}'")
    return None


def annotate_frame(frame: np.ndarray, timestep: int, action_type: str, apd_step: str) -> np.ndarray:
    """
    Annotate frame with timestep, action type, and APD step in the top-right corner.
    
    Args:
        frame: (H, W, 3) RGB image
        timestep: Current timestep number
        action_type: Action type ("move", "pick", "place")
        apd_step: APD plan step description
    
    Returns:
        Annotated frame
    """
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    
    # Define text properties (scaled to image size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font based on image width (for 256px width, this gives ~0.37)
    font_scale = width / 700.0
    thickness = 1
    # Scale line spacing based on image height
    line_spacing = int(height / 16.0)
    
    # Define colors (BGR format)
    bg_color = (0, 0, 0)  # Black background
    text_color = (255, 255, 255)  # White text
    action_colors = {
        "move": (255, 200, 0),  # Cyan
        "pick": (0, 255, 0),   # Green
        "place": (0, 100, 255)  # Orange
    }
    action_color = action_colors.get(action_type, text_color)
    
    # Prepare text lines
    lines = [
        (f"t={timestep}", text_color),
        (f"{action_type.upper()}", action_color),
    ]
    
    # Split APD step into multiple lines if too long
    max_apd_width = 45  # characters
    apd_words = apd_step.split()
    apd_lines = []
    current_line = ""
    
    for word in apd_words:
        if len(current_line) + len(word) + 1 <= max_apd_width:
            current_line += (word + " ")
        else:
            if current_line:
                apd_lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        apd_lines.append(current_line.strip())
    
    # Add APD lines
    for apd_line in apd_lines:
        lines.append((apd_line, text_color))
    
    # Calculate text box dimensions
    max_text_width = 0
    text_heights = []
    
    for text, _ in lines:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)
        text_heights.append(text_height + baseline)
    
    # Define text box position (top-right corner, scaled to image size)
    padding = int(width / 50.0)  # ~5px for 256px width
    box_width = max_text_width + 2 * padding
    box_height = len(lines) * line_spacing + padding
    
    margin = int(width / 50.0)  # ~5px for 256px width
    box_x1 = width - box_width - margin
    box_y1 = margin
    box_x2 = width - margin
    box_y2 = box_y1 + box_height
    
    # Draw semi-transparent background
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
    
    # Draw border
    cv2.rectangle(frame_copy, (box_x1, box_y1), (box_x2, box_y2), text_color, 1)
    
    # Draw text
    y_offset = box_y1 + padding + int(height / 20.0)  # ~13px for 256px height
    for text, color in lines:
        cv2.putText(frame_copy, text, (box_x1 + padding, y_offset), 
                   font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += line_spacing
    
    return frame_copy


def replay_episode(
    env,
    actions: np.ndarray,
    initial_state: np.ndarray,
    episode_labels: Optional[Dict] = None,
    num_steps_wait: int = 10
) -> Tuple[List[np.ndarray], bool]:
    """
    Replay episode with expert actions and annotate frames.
    
    Args:
        env: LIBERO environment
        actions: (T, 7) expert action sequence
        initial_state: Initial state for the environment
        episode_labels: Timestep labels dictionary
        num_steps_wait: Number of wait steps at beginning
    
    Returns:
        (annotated_frames, success)
        - annotated_frames: List of annotated RGB frames
        - success: Whether episode completed successfully
    """
    # Reset environment
    env.reset()
    
    # Set initial state
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    
    annotated_frames = []
    success = False
    T = len(actions)
    
    # Get timestep labels if available
    timestep_labels = None
    if episode_labels is not None:
        timestep_labels = episode_labels.get('timestep_labels', None)
    
    # Run episode with wait steps
    for t in range(T + num_steps_wait):
        # Wait steps: do nothing
        if t < num_steps_wait:
            action = np.array([0, 0, 0, 0, 0, 0, -1])  # Dummy action
        else:
            # Get expert action
            action_idx = t - num_steps_wait
            if action_idx < T:
                action = actions[action_idx]
            else:
                # Shouldn't reach here
                break
        
        # Get frame before step
        img = get_libero_image(obs)
        
        # Annotate frame
        if timestep_labels is not None and t >= num_steps_wait:
            action_idx = t - num_steps_wait
            if action_idx < len(timestep_labels):
                label = timestep_labels[action_idx]
                timestep_num = label['timestep']
                action_type = label['action']
                apd_step = label['APD_step']
                
                img = annotate_frame(img, timestep_num, action_type, apd_step)
        
        annotated_frames.append(img)
        
        # Execute action
        obs, reward, done, info = env.step(action.tolist())
        
        if done:
            success = True
            logger.info(f"Episode completed successfully at timestep {t}")
            break
    
    return annotated_frames, success


def save_video(frames: List[np.ndarray], 
              output_path: str, 
              fps: int = 30) -> None:
    """
    Save frames as MP4 video.
    
    Args:
        frames: List of RGB frames
        output_path: Output video file path
        fps: Frames per second
    """
    logger.info(f"Saving video to {output_path}")
    
    video_writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        video_writer.append_data(frame)
    video_writer.close()
    
    logger.info(f"Video saved: {output_path}")


def get_initial_state_for_episode(
    task_suite,
    task_id: int,
    episode_idx: int
) -> np.ndarray:
    """
    Get initial state for specified episode.
    
    Args:
        task_suite: LIBERO task suite
        task_id: Task ID
        episode_idx: Episode index
    
    Returns:
        Initial state array
    """
    initial_states = task_suite.get_task_init_states(task_id)
    
    if episode_idx < len(initial_states):
        return initial_states[episode_idx]
    else:
        logger.warning(f"Episode index {episode_idx} exceeds available initial states, using last available")
        return initial_states[-1]


def main(args):
    """Main function to replay expert episode and generate annotated video."""
    
    logger.info("="*60)
    logger.info("LIBERO Expert Episode Replay")
    logger.info("="*60)
    
    # Load substep labels
    substep_labels = load_substep_labels(args.substep_labels_path)
    
    # Load RLDS episode
    episode = load_rlds_episode(args.rlds_data_dir, args.suite, args.episode_idx)
    if episode is None:
        logger.error("Failed to load episode, exiting")
        return
    
    # Extract episode data
    actions, instruction, task_name = extract_episode_data(episode)
    logger.info(f"Task instruction: {instruction}")
    logger.info(f"Task name: {task_name}")
    logger.info(f"Total timesteps: {len(actions)}")
    
    # Find episode labels
    episode_labels = find_episode_labels(substep_labels, args.suite, task_name, args.episode_idx)
    if episode_labels is None:
        logger.warning("No substep labels found for this episode, will create video without annotations")
    else:
        logger.info(f"Found substep labels with {len(episode_labels.get('timestep_labels', []))} timesteps")
    
    # Initialize LIBERO task suite and environment
    suite_name_for_benchmark = args.suite.replace('_no_noops', '')
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name_for_benchmark]()
    
    # Find task ID by matching instruction
    task_id = None
    for tid in range(task_suite.n_tasks):
        task = task_suite.get_task(tid)
        if task.language.lower().strip() == instruction:
            task_id = tid
            break
    
    if task_id is None:
        logger.error(f"Could not find matching task for instruction: {instruction}")
        return
    
    logger.info(f"Matched to task ID: {task_id}")
    
    # Get task and initialize environment
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, model_family="openvla", resolution=args.env_img_res)
    
    # Get initial state
    initial_state = get_initial_state_for_episode(task_suite, task_id, args.episode_idx)
    
    # Replay episode and generate annotated video
    logger.info("Replaying episode with expert actions...")
    frames, success = replay_episode(
        env,
        actions,
        initial_state,
        episode_labels,
        num_steps_wait=args.num_steps_wait
    )
    
    logger.info(f"Episode replay complete. Success: {success}, Total frames: {len(frames)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    output_filename = f"{args.suite}__episode_{args.episode_idx}__{task_name}__success_{success}.mp4"
    output_path = os.path.join(args.output_dir, "expert_replay.mp4")
    
    # Save video
    save_video(frames, output_path, fps=args.fps)
    
    logger.info("="*60)
    logger.info("Replay complete!")
    logger.info(f"Video saved to: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay expert LIBERO episode and generate annotated video"
    )
    
    parser.add_argument(
        "--rlds_data_dir",
        type=str,
        required=True,
        help="Path to RLDS dataset directory (e.g., /path/to/modified_libero_rlds)"
    )
    
    parser.add_argument(
        "--substep_labels_path",
        type=str,
        required=True,
        help="Path to substep_labels_output.json file"
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Task suite name (e.g., libero_spatial_no_noops)"
    )
    
    parser.add_argument(
        "--episode_idx",
        type=int,
        required=True,
        help="Episode index to replay"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./expert_replay_videos",
        help="Output directory for videos"
    )
    
    parser.add_argument(
        "--env_img_res",
        type=int,
        default=256,
        help="Resolution for environment images"
    )
    
    parser.add_argument(
        "--num_steps_wait",
        type=int,
        default=10,
        help="Number of wait steps at beginning"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second"
    )
    
    args = parser.parse_args()
    
    main(args)

