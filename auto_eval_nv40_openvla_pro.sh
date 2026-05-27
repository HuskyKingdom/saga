#!/bin/bash
echo "Running Evaluations Automatically ------------------------------"
FARM_USER=sgyson10
FILE_PATH="experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml"
CKPT_SPATIAL="openvla/openvla-7b-finetuned-libero-spatial"
CKPT_OBJECT="openvla/openvla-7b-finetuned-libero-object"
CKPT_GOAL="openvla/openvla-7b-finetuned-libero-goal"
CKPT_10="openvla/openvla-7b-finetuned-libero-10"
CKPT_apd="/home/yuhang/Warehouse/Yuhangworkspace/openvla-oft-yhs/ckpt/landmarked_ckpoints/apd_regressive_200k"
TASK_LABEL_PREFIX="openvla"
EVAL_SCRIPT="experiments/robot/libero/run_libero_pro_eval.py"
USE_PROPRIO=False
USE_L1_REGRESSION=False
SAVE_VIDEO=False
NUM_IMAGES_IN_INPUT=1
USE_BDDL_LANGUAGE=True
AUTO_REGRESSION=False



echo "Evaluating SUBSTEP ------------------------------"

# Reset all evaluation config flags to false before starting
bash experiments/robot/libero/LIBERO-PRO/reset_eval_config.sh $FILE_PATH


# lan
sed -i 's/use_environment: true/use_environment: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_language: false/use_language: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_GOAL \
  --task_suite_name libero_goal --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_lan --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION


python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OBJECT \
  --task_suite_name libero_object --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_lan --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_SPATIAL \
  --task_suite_name libero_spatial --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_lan --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_10 \
  --task_suite_name libero_10 --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_lan --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION


# object
sed -i 's/use_language: true/use_language: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_object: false/use_object: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OBJECT \
  --task_suite_name libero_object --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_object --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_GOAL \
  --task_suite_name libero_goal --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_object --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_SPATIAL \
  --task_suite_name libero_spatial --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_object --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_10 \
  --task_suite_name libero_10 --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_object --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION


# swap
sed -i 's/use_object: true/use_object: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_swap: false/use_swap: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml


python $EVAL_SCRIPT  \
  --pretrained_checkpoint $CKPT_OBJECT \
  --task_suite_name libero_object --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_swap --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_GOAL \
  --task_suite_name libero_goal --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_swap --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT  \
  --pretrained_checkpoint $CKPT_SPATIAL \
  --task_suite_name libero_spatial --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_swap --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_10 \
  --task_suite_name libero_10 --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_swap --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language False --auto_regression $AUTO_REGRESSION


# task
sed -i 's/use_swap: true/use_swap: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
sed -i 's/use_task: false/use_task: true/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_OBJECT \
  --task_suite_name libero_object --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_object --task_label ${TASK_LABEL_PREFIX}_object_task --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_GOAL \
  --task_suite_name libero_goal --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_goal --task_label ${TASK_LABEL_PREFIX}_goal_task --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_SPATIAL \
  --task_suite_name libero_spatial --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_spatial --task_label ${TASK_LABEL_PREFIX}_spatial_task --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION

python $EVAL_SCRIPT \
  --pretrained_checkpoint $CKPT_10 \
  --task_suite_name libero_10 --save_video $SAVE_VIDEO \
  --num_trials_per_task 50 --evaluation_config_path experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml  --unnorm_key libero_10 --task_label ${TASK_LABEL_PREFIX}_10_task --use_proprio $USE_PROPRIO --use_l1_regression $USE_L1_REGRESSION --num_images_in_input $NUM_IMAGES_IN_INPUT --use_bddl_language $USE_BDDL_LANGUAGE --auto_regression $AUTO_REGRESSION

sed -i 's/use_task: true/use_task: false/' experiments/robot/libero/LIBERO-PRO/evaluation_config.yaml
