#!/bin/bash
# InfoBot 200K Evaluation Script - Following auto_eval_nv40_plus structure

cd /home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs

# Setup environment
source /home/yuhang/Warehouse/Yuhangworkspace/miniconda3/etc/profile.d/conda.sh
conda activate vla_pro
export PYTHONPATH=$PYTHONPATH:/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/experiments/robot/libero/LIBERO-PRO

# Checkpoint path
CKPT="/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b8+lr-0.0002+lora-r32+infobot-cross_attn-beta0.1--image_aug--substep--infobot_v2_stable--200000_chkpt"
LABEL="infobot_200k"

echo "========================================"
echo "InfoBot 200K Evaluation"
echo "Checkpoint: $CKPT"
echo "Start: $(date)"
echo "========================================"

# Common args for InfoBot
COMMON="--model_family openvla \
  --vla_path openvla/openvla-7b \
  --pretrained_checkpoint $CKPT \
  --use_infobot True \
  --infobot_bottleneck_type cross_attn \
  --infobot_bottleneck_dim 256 \
  --infobot_num_tokens 8 \
  --use_l1_regression True \
  --num_images_in_input 2 \
  --center_crop True \
  --use_proprio False \
  --num_trials_per_task 50 \
  --seed 7 \
  --save_video False \
  --e_decoding False"

# 1. LIBERO-Spatial
echo ""
echo "=== [1/4] LIBERO-Spatial ==="
echo "Start: $(date)"
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON \
  --task_suite_name libero_spatial \
  --unnorm_key libero_spatial_no_noops \
  --run_id_note "${LABEL}_spatial"

# 2. LIBERO-Object
echo ""
echo "=== [2/4] LIBERO-Object ==="
echo "Start: $(date)"
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON \
  --task_suite_name libero_object \
  --unnorm_key libero_object_no_noops \
  --run_id_note "${LABEL}_object"

# 3. LIBERO-Goal
echo ""
echo "=== [3/4] LIBERO-Goal ==="
echo "Start: $(date)"
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON \
  --task_suite_name libero_goal \
  --unnorm_key libero_goal_no_noops \
  --run_id_note "${LABEL}_goal"

# 4. LIBERO-10 (Long-horizon)
echo ""
echo "=== [4/4] LIBERO-10 ==="
echo "Start: $(date)"
python experiments/robot/libero/run_libero_pro_eval_javas.py \
  $COMMON \
  --task_suite_name libero_10 \
  --unnorm_key libero_10_no_noops \
  --run_id_note "${LABEL}_10"

echo ""
echo "========================================"
echo "Evaluation Complete: $(date)"
echo "========================================"

# Summary - extract success rates
echo ""
echo "=== Results Summary ==="
for suite in spatial object goal 10; do
    KEY="libero_${suite}"
    [ "$suite" = "10" ] && KEY="libero_10"
    
    # Find the latest log file
    LOG=$(ls -t experiments/logs/EVAL-${KEY}*${LABEL}_${suite}*.txt 2>/dev/null | head -1)
    if [ -f "$LOG" ]; then
        RATE=$(grep "Overall success rate" "$LOG" | tail -1 | awk '{print $NF}')
        echo "${KEY}: ${RATE:-N/A}"
    else
        echo "${KEY}: Log not found"
    fi
done

echo ""
echo "All logs saved to: experiments/logs/"
