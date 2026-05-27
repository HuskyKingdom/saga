#!/bin/bash
# Reset all evaluation config flags to false before starting evaluation.
# Usage: bash reset_eval_config.sh [path/to/evaluation_config.yaml]

CONFIG_FILE="${1:-experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[CONFIG] Warning: config file not found: $CONFIG_FILE"
    exit 1
fi

sed -i 's/use_environment: true/use_environment: false/' "$CONFIG_FILE"
sed -i 's/use_language: true/use_language: false/' "$CONFIG_FILE"
sed -i 's/use_task: true/use_task: false/' "$CONFIG_FILE"
sed -i 's/use_swap: true/use_swap: false/' "$CONFIG_FILE"
sed -i 's/use_object: true/use_object: false/' "$CONFIG_FILE"

echo "[CONFIG] All evaluation flags reset to false: $CONFIG_FILE"
