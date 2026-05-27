#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide checkpoint path as argument"
    echo "Usage: $0 <ckpt_path>"
    exit 1
fi

# Get the input ckpt_path
CKPT_PATH="$1"

# Check if target path exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "Error: Target path '$CKPT_PATH' does not exist or is not a directory"
    exit 1
fi

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# Define the list of files to copy
FILES=(
    "${SCRIPT_DIR}/verl/utils/vla_utils/openvla_oft/configuration_prismatic.py"
    "${SCRIPT_DIR}/verl/utils/vla_utils/openvla_oft/constants.py"
    "${SCRIPT_DIR}/verl/utils/vla_utils/openvla_oft/modeling_prismatic.py"
    "${SCRIPT_DIR}/verl/utils/vla_utils/openvla_oft/processing_prismatic.py"
    "${SCRIPT_DIR}/verl/utils/vla_utils/openvla_oft/train_utils.py"
)

# Copy files

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp -f "$file" "$CKPT_PATH/"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully overwrite: $filename"
        else
            echo "✗ Failed to overwrite: $filename"
        fi
    else
        echo "✗ File not found: $file"
    fi
done

echo "File overwrite completed!"

# Also patch the HuggingFace transformers_modules cache so the loaded model
# uses the updated code (HF caches remote-code files separately from weights).
CKPT_NAME=$(basename "$CKPT_PATH")
HF_MODULES_CACHE="${SCRIPT_DIR}/.cache/huggingface/modules/transformers_modules"
if [ -d "$HF_MODULES_CACHE" ]; then
    for CACHE_DIR in "$HF_MODULES_CACHE"/*/; do
        CACHE_NAME=$(basename "$CACHE_DIR")
        # Match by checkpoint name substring (handles both exact and hash-suffixed dirs)
        if [[ "$CACHE_NAME" == *"$CKPT_NAME"* ]]; then
            echo "Found HF modules cache: $CACHE_DIR"
            for file in "${FILES[@]}"; do
                if [ -f "$file" ]; then
                    filename=$(basename "$file")
                    cp -f "$file" "$CACHE_DIR/"
                    echo "✓ Patched cache: $CACHE_DIR$filename"
                fi
            done
        fi
    done
fi