export SFT_MODEL_PATH="exp_out/openvla-oft-rl/oft_plus-rl-scope/actor/global_step_24"
export CKPT_PATH="./exp_out"
export DATASET_NAME="libero_4_task_suites"
export EXPERIMENT_NAME="scope-rl-aggresive"
export PROJECT_NAME="scope-rl-aggresive"
export NUM_GPUS=8
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Use autoregressive generation (for SFT models trained with use_l1_regression=False)
export USE_AUTOREGRESSIVE="False"
# Randomly swap two objects' (x,y) positions at episode start to encourage instruction grounding
export SWAP_OBJECTS="True"
export SWAP_DISTANCE_START="0.14"     # metres; max allowed swap distance at curriculum start
export SWAP_DISTANCE_END="0.40"       # metres; max allowed swap distance at curriculum end
export SWAP_CURRICULUM_STEPS="12500"      # training steps to reach max distance (0 = no curriculum)


# Distance reward: Gaussian kernel on min gripper-to-target distance, mapped to [0,1]
# Ensure dist_reward_coef * 1.0 << verifier reward (5.0), so success always dominates
export DIST_REWARD_COEF="0.3"         # weight; 0.3 << 5.0 (success reward)
export DIST_REWARD_SIGMA="0.05"       # Gaussian width in metres (~5 cm)

export TRAINER_INITIAL_GLOBAL_STEPS="24"

export DATA_TRAIN_BATCH_SIZE="8"
export ACTOR_PPO_MINI_BATCH_SIZE="32"   # must be <= DATA_TRAIN_BATCH_SIZE * n_samples (8*4=32)
export ACTOR_TRAJ_MINI_BATCH_SIZE="8"   # must be <= DATA_TRAIN_BATCH_SIZE

sbatch examples/run_openvla_oft_scope.sh
