export SFT_MODEL_PATH="exp_out/openvla-oft-rl/grpo-liberospatial-run1/actor/global_step_24"
export CKPT_PATH="./exp_out"
export DATASET_NAME="libero_4_task_suites"
export EXPERIMENT_NAME="oft_plus-rl"
export PROJECT_NAME="openvla-oft-rl"
export NUM_GPUS=8
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Use autoregressive generation (for SFT models trained with use_l1_regression=False)
export USE_AUTOREGRESSIVE="False"


export DATA_TRAIN_BATCH_SIZE=8
export ACTOR_PPO_MINI_BATCH_SIZE=32   # must be <= DATA_TRAIN_BATCH_SIZE * n_samples (8*4=32)
export ACTOR_TRAJ_MINI_BATCH_SIZE=8   # must be <= DATA_TRAIN_BATCH_SIZE
export TRAINER_TOTAL_EPOCHS=75

sbatch examples/run_openvla_oft_plus_rl_libero.sh
