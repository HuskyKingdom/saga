#!/bin/bash

# Example script to replay expert episodes from LIBERO RLDS dataset
# This script demonstrates how to use run_expert_replay.py

# Configuration
RLDS_DATA_DIR="/path/to/modified_libero_rlds"  # TODO: Update this path
SUBSTEP_LABELS_PATH="./substep_labels_output.json"  # TODO: Update this path
OUTPUT_DIR="./expert_replay_videos"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "LIBERO Expert Episode Replay Examples"
echo "========================================"
echo ""

# Example 1: Replay first episode from libero_spatial
echo "Example 1: Replaying libero_spatial episode 0..."
python run_expert_replay.py \
    --rlds_data_dir "$RLDS_DATA_DIR" \
    --substep_labels_path "$SUBSTEP_LABELS_PATH" \
    --suite libero_spatial_no_noops \
    --episode_idx 0 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Example 1 complete!"
echo ""

# Example 2: Replay fifth episode from libero_object
echo "Example 2: Replaying libero_object episode 5..."
python run_expert_replay.py \
    --rlds_data_dir "$RLDS_DATA_DIR" \
    --substep_labels_path "$SUBSTEP_LABELS_PATH" \
    --suite libero_object_no_noops \
    --episode_idx 5 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Example 2 complete!"
echo ""

# Example 3: Replay with custom settings
echo "Example 3: Replaying with custom FPS and resolution..."
python run_expert_replay.py \
    --rlds_data_dir "$RLDS_DATA_DIR" \
    --substep_labels_path "$SUBSTEP_LABELS_PATH" \
    --suite libero_goal_no_noops \
    --episode_idx 0 \
    --output_dir "$OUTPUT_DIR" \
    --fps 60 \
    --env_img_res 512

echo ""
echo "Example 3 complete!"
echo ""

# Example 4: Batch replay multiple episodes
echo "Example 4: Batch replaying multiple episodes from libero_spatial..."
for idx in 0 1 2 3 4; do
    echo "  Processing episode $idx..."
    python run_expert_replay.py \
        --rlds_data_dir "$RLDS_DATA_DIR" \
        --substep_labels_path "$SUBSTEP_LABELS_PATH" \
        --suite libero_spatial_no_noops \
        --episode_idx $idx \
        --output_dir "$OUTPUT_DIR" \
        > /dev/null 2>&1  # Suppress output for batch processing
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Episode $idx completed successfully"
    else
        echo "  ✗ Episode $idx failed"
    fi
done

echo ""
echo "Example 4 complete!"
echo ""

echo "========================================"
echo "All examples finished!"
echo "Videos saved to: $OUTPUT_DIR"
echo "========================================"

