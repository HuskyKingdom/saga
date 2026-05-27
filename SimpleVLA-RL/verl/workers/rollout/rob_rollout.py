# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import os
import torch

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
import sys
import importlib

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
import verl.utils.torch_functional as verl_F
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor

from verl.utils.libero_utils import save_rollout_video
try:
    from verl.utils.libero_utils import (
        get_libero_env, get_libero_dummy_action, get_libero_image, 
        get_libero_wrist_image, quat2axisangle, normalize_gripper_action, 
        invert_gripper_action
    )
except ImportError as e:
    print(f"Warning : can't import libero: {e}")
    
from verl.utils.vla_utils.openvla_oft.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import deque
import random
import yaml
from pathlib import Path

import threading
import queue
import gc
from collections import defaultdict
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from codetiming import Timer

# For Libero multiprocessing
import multiprocessing
from multiprocessing import get_context as mp_get_context
from libero_env_worker import env_worker as _libero_env_worker

__all__ = ['RobHFRollout']

# Environment initialization lock for Robotwin
_ENV_INIT_LOCK = threading.Lock()


def _compute_swap_max_distance(global_steps: int, config) -> float:
    """Linear curriculum: allowed swap distance grows from min to max over training.

    Config fields (all under actor_rollout_ref.rollout):
        swap_min_distance     : metres at step 0           (default 0.05)
        swap_max_distance     : metres at curriculum end   (default 1e9 = no limit)
        swap_curriculum_steps : steps to reach max         (default 0 = no curriculum)
    """
    min_d = getattr(config, 'swap_min_distance', 0.05)
    max_d = getattr(config, 'swap_max_distance', 1e9)
    curriculum_steps = getattr(config, 'swap_curriculum_steps', 0)
    if curriculum_steps <= 0:
        return max_d
    progress = min(global_steps / curriculum_steps, 1.0)
    return min_d + progress * (max_d - min_d)

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.
    """
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image):
    batch_size = 1
    crop_scale = 0.9

    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")
    return image

# ================ Robotwin-specific functions ================

def normalize_proprio(proprio, norm_stats):
    """Normalize proprioception data for Robotwin."""
    if ACTION_PROPRIO_NORMALIZATION_TYPE == "bounds":
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == "bounds_q99":
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")
    
    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized_proprio



def get_robotwin2_task(task_name, config):
    """Get robotwin 2.0 task"""
    robotwin2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'envs', 'robotwin2')
    if robotwin2_path not in sys.path:
        sys.path.append(robotwin2_path)
        
    robotwin2_utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'envs', 'robotwin2', "description", "utils")
    if robotwin2_utils_path not in sys.path:
        sys.path.append(robotwin2_utils_path)
    
    from envs import CONFIGS_PATH
    
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit(f"No Task: {task_name}")
    
    task_config = config.get('twin2_task_config', 'demo_randomized')
    config_file = os.path.join(robotwin2_path, f"task_config/{task_config}.yml")
    
    with open(config_file, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    args['task_config'] = task_config
    args['ckpt_setting'] = config.get('twin2_ckpt_setting', 'demo_randomized')
    
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file
    
    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        return embodiment_args
    
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")
    
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]
    
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["render_freq"] = 0
    args['instruction_type'] = config.get('twin2_instruction_type', 'unseen')
    
    return env_instance, args

def encode_obs(observation):
    """Post-Process Observation for robotwin 2.0"""
    return observation

class RobotwinEnvWrapper:
    """Thread-safe wrapper for Robotwin environment (supports both 1.0 and 2.0)"""
    def __init__(self, task_name, trial_id, trial_seed, config, version="1.0"):
        self.task_name = task_name
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.config = config
        self.version = version
        self.env = None
        self.args = None
        self.active = True
        self.complete = False
        self.finish_step = 0
        self.lock = threading.Lock()
        self.instruction = None
        
    def initialize(self):
        """Initialize the environment"""
        with _ENV_INIT_LOCK:
            with self.lock:
                try:
                    if self.version == "1.0":
                        print("RobotWin 2.0 fully encompasses RobotWin 1.0, therefore we prioritize support for RobotWin 2.0")
                        raise ValueError
                    else:  # 2.0
                        self.env, self.args = get_robotwin2_task(self.task_name, self.config)
                        self.env.setup_demo(now_ep_num=self.trial_id, seed=self.trial_seed, is_test=True, **self.args)
                        episode_info_list = [self.env.get_info()]
                except Exception as e:
                    print(f"****** IN thread: setup_demo ERROR {e} ******", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.env, self.args = get_robotwin2_task(self.task_name, self.config)
                    self.env.setup_demo(now_ep_num=self.trial_id, seed=self.trial_seed, is_test=True, **self.args)
                    episode_info_list = [self.env.get_info()]
                
                
                from generate_episode_instructions import generate_episode_descriptions
                results = generate_episode_descriptions(self.task_name, episode_info_list, 1, seed=self.trial_id)
                self.instruction = np.random.choice(results[0][self.args["instruction_type"]])
                self.env.set_instruction(instruction=self.instruction)
                
    def get_obs(self):
        """Get observation from environment"""
        with self.lock:
            try:
                geted_obs = self.env.get_obs()
                return geted_obs
            except Exception as e:
                print(f"****** IN thread: get_obs ERROR {e} ******", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                geted_obs = self.env.get_obs()
                return geted_obs
    
    def get_instruction(self):
        """Get instruction for the task"""
        with self.lock:
            
            return self.env.get_instruction()
            
    def step(self, action):
        """Execute action in environment"""
        with self.lock:
            try:
                
                self.env.take_action(action)
                done = self.env.eval_success
                    
            except Exception as e:
                done = False
                error_msg = f"****** action execution ERROR: {type(e).__name__}: {str(e)} ******"
                print(error_msg, flush=True)
                traceback.print_exc()
                
            try:
                obs = self.env.get_obs()
                obs = encode_obs(obs)
            except Exception as e:
                print(f"****** env.get_obs ERROR {e} ******", flush=True)
                obs = None
                
            self.finish_step += action.shape[0]
            
            if done or self.finish_step >= self.env.step_lim:
                self.active = False
                self.complete = done
            
            return obs, done
            
    def close(self):
        """Close the environment"""
        with self.lock:
            if self.env is not None:
                try:
                    self.env.close_env(clear_cache=True)
                except Exception as e:
                    print(f"******IN env.close ERROR {e} ******", flush=True)

# ================ Libero-specific functions ================

def env_worker(task_name, task_id, trial_id, config, input_queue, output_queue, is_valid, global_steps, max_steps):
    """Worker process for Libero environments"""
    from libero.libero import benchmark
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    initial_state = initial_states[trial_id]
    
    env = None
    import sys, traceback
    _MAX_RETRY = 5
    for _attempt in range(_MAX_RETRY):
        try:
            env, task_description = get_libero_env(task, config.model_family, resolution=256)
            break
        except Exception as _exc:
            print(f"*** env initialization failed (attempt {_attempt+1}/{_MAX_RETRY}): "
                  f"{type(_exc).__name__}: {_exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            if env is not None:
                try:
                    env.close()
                except Exception as e:
                    print(f"error when close the env: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            print("gc collect finish")
    else:
        raise RuntimeError(
            f"get_libero_env failed after {_MAX_RETRY} attempts "
            f"(task_id={task_id}, trial_id={trial_id}). See traceback above."
        )
    
    env.reset()
    obs = env.set_init_state(initial_state)
    
    t = 0
    valid_images = []
    while t < config.num_steps_wait:
        obs, _, _, _ = env.step(get_libero_dummy_action(config.model_family))
        t += 1
        
    if is_valid:
        img = obs["agentview_image"][::-1, ::-1]
        valid_images.append(img)
    
    output_queue.put({
        'type': 'init',
        'obs': obs,
        "task_description": task_description,
        'valid_images': valid_images.copy(),
        'task_file_name': f"{task_name}_task_{task_id}_trial_{trial_id}",
        'active': True,
        'complete': False,
        'finish_step': 0
    })
    
    active = True
    complete = False
    finish_step = 0
    
    while True:
        action = input_queue.get()
        if action is None:
            env.close()
            output_queue.put({'type': 'terminate'})
            break
        
        step_images = []
        for i in range(len(action)):
            a = action[i]
            normalized_action = normalize_gripper_action(a, binarize=True)
            inverted_action = invert_gripper_action(normalized_action)
            obs, reward, done, info = env.step(inverted_action.tolist())
            
            if is_valid:
                img = obs["agentview_image"][::-1, ::-1]
                step_images.append(img)
            
            finish_step += 1
            if done or finish_step >= max_steps:
                active = False
                complete = done
                break
        
        output_data = {
            'type': 'step',
            'obs': obs,
            'active': active,
            'complete': complete,
            'finish_step': finish_step,
            'valid_images': step_images.copy() if is_valid else []
        }
        output_queue.put(output_data)

# ================ Main Rollout Class ================

class RobHFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.max_steps = {
            "libero_spatial": 512,
            "libero_object": 512,
            "libero_goal": 512,
            "libero_10": 512,
            "libero_90": 512,
            "libero_4_task_suites": 512,
            "robotwin2_click_bell": 200,
            "robotwin2_move_can_pot": 200,
            "robotwin2_place_phone_stand": 200,
            "robotwin2_place_a2b_left": 200,
            "robotwin2_place_a2b_right": 200,
            "robotwin2_handover_mic": 200,
            "robotwin2_pick_dual_bottles": 100,
            "robotwin2_lift_pot": 200,
            "robotwin2_put_bottles_dustbin": 800,
            "robotwin2_stack_blocks_two": 400,
            "robotwin2_stack_bowls_two": 400,
            "robotwin2_handover_block": 400,
            "robotwin2_place_empty_cup": 200,
            "robotwin2_shake_bottle": 75,
            "robotwin2_move_stapler_pad": 200,
            "robotwin2_place_container_plate": 150,
            "robotwin2_blocks_ranking_rgb": 600,
            "robotwin2_beat_block_hammer": 200,
            "robotwin2_place_mouse_pad": 200,
            "robotwin2_place_shoe": 250,
            "robotwin2_move_pillbottle_pad": 200,
        }
        self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
        self.vla_preprocess()

        # SAGA: load plan configs if enabled
        self.saga_plan_configs = {}
        saga_cfg = getattr(config, 'saga_apd_plans_path', None)
        if saga_cfg:
            from verl.utils.saga import load_saga_plan_configs
            self.saga_plan_configs = load_saga_plan_configs(saga_cfg)
            print(f"[SAGA] Loaded plan configs for {len(self.saga_plan_configs)} tasks")

        # Setup execution pool based on task suite
        if "robotwin" in self.config.task_suite_name:
            self.env_thread_pool = ThreadPoolExecutor(max_workers=16)
            self.robotwin_version = self._detect_robotwin_version()
        
    def _detect_robotwin_version(self):
        """Detect which version of robotwin to use based on config"""
        if hasattr(self.config, 'robotwin_version'):
            return self.config.robotwin_version
        elif 'robotwin2' in self.config.task_suite_name:
            return "2.0"
        else:
            print("RobotWin 2.0 fully encompasses RobotWin 1.0, therefore we prioritize support for RobotWin 2.0")
            raise ValueError
        
    def vla_preprocess(self):
        if self.config.vla in ["openvla", "openvla-oft"]:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        
        if self.config.vla in ["openvla-oft"]:
            if "libero" in self.config.task_suite_name:
                # Standard single-suite fallback: libero_10 → libero_10_no_noops
                if self.config.unnorm_key not in self.module.norm_stats and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats:
                    self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
                # Combined-suite fallback: libero_4_task_suites / libero_4_task_suites_no_noops not found,
                # so scan for any constituent libero_*_no_noops key in norm_stats.
                # All LIBERO suites share the same Franka Panda action space, so any suite's stats work.
                if self.config.unnorm_key not in self.module.norm_stats:
                    _libero_suite_candidates = ["libero_spatial_no_noops", "libero_object_no_noops",
                                                "libero_goal_no_noops", "libero_10_no_noops",
                                                "libero_90_no_noops",
                                                "libero_spatial", "libero_object",
                                                "libero_goal", "libero_10", "libero_90"]
                    for _candidate in _libero_suite_candidates:
                        if _candidate in self.module.norm_stats:
                            print(f"[vla_preprocess] unnorm_key '{self.config.unnorm_key}' not found; "
                                  f"falling back to '{_candidate}' for combined LIBERO suite.")
                            self.config.unnorm_key = _candidate
                            break
            elif "robotwin" in self.config.task_suite_name:
                self.config.unnorm_key = self.config.unnorm_key.removeprefix("robotwin_").removeprefix("robotwin2_")
            assert self.config.unnorm_key in self.module.norm_stats, f"Action un-norm key {self.config.unnorm_key} not found in VLA `norm_stats`!"

    def generate_sequences(self, prompts):
        batch_size = prompts.batch.batch_size[0]
        
        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)
            
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output
    
    def process_input(self, inputs: list, task_descriptions: list):
        """Unified input processing for both Robotwin and Libero"""
        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        if self.config.use_proprio:
            batchdata["proprio"] = []
        
        for i in range(len(inputs)):
            input_data = inputs[i]
            task_description = task_descriptions[i]
            
            # Process main image
            image = Image.fromarray(input_data["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature = self.processor(prompt, image)
            
            pixel_values_list = [batch_feature["pixel_values"]]
            
            # Process additional images (wrist cameras)
            if "robotwin" in self.config.task_suite_name:
                # Robotwin may have multiple wrist images
                for key in input_data:
                    if "wrist" in key and isinstance(input_data[key], np.ndarray):
                        wrist_image = Image.fromarray(input_data[key]).convert("RGB")
                        if self.config.center_crop:
                            wrist_image = center_crop_image(wrist_image)
                        wrist_batch_feature = self.processor(prompt, wrist_image)
                        pixel_values_list.append(wrist_batch_feature["pixel_values"])
            else:
                # Libero has single wrist image
                if "wrist_image" in input_data:
                    wrist_image = Image.fromarray(input_data["wrist_image"]).convert("RGB")
                    if self.config.center_crop:
                        wrist_image = center_crop_image(wrist_image)
                    wrist_batch_feature = self.processor(prompt, wrist_image)
                    pixel_values_list.append(wrist_batch_feature["pixel_values"])
            
            batch_feature["pixel_values"] = torch.cat(pixel_values_list, dim=1)
            
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                if self.config.vla in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
                    )
            
            batchdata["input_ids"].append(input_ids)
            batchdata["attention_mask"].append(attention_mask)
            batchdata["pixel_values"].append(pixel_values)
            
            # Process proprioception
            if self.config.use_proprio:
                if "robotwin" in self.config.task_suite_name:
                    proprio = input_data["state"]
                    proprio_norm_stats = self.module.norm_stats[self.config.unnorm_key]["proprio"]
                    proprio = normalize_proprio(proprio, proprio_norm_stats)
                    batchdata["proprio"].append(torch.from_numpy(proprio))
                elif "state" in input_data:
                    # Libero: pass raw state directly to proprio_projector
                    proprio = input_data["state"].astype(np.float32)
                    batchdata["proprio"].append(torch.from_numpy(proprio))
        
        device = torch.device('cuda')
        
        # Padding and device placement
        if self.config.vla in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            batchdata["input_ids"] = pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
            
            padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            assert torch.all(padding_mask == batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int()
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(device)
            
            if self.config.use_proprio and "proprio" in batchdata and len(batchdata["proprio"]) > 0:
                batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(device)
                
            assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id))
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

        return batchdata
    
    def _generate_minibatch(self, prompts):
        """Generate minibatch - routes to appropriate implementation based on task suite"""
        if "robotwin" in self.config.task_suite_name:
            return self._generate_minibatch_robotwin(prompts)
        else:
            return self._generate_minibatch_libero(prompts)
    
    def _generate_minibatch_robotwin(self, prompts):
        """Generate minibatch for Robotwin using threading"""
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        trial_seed = prompts.batch['trial_seed'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps.get(self.config.task_suite_name, 800)
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        
        # Create environment wrappers
        env_wrappers = []
        for idx in range(batch_size):
            task_name = task_suite_name[idx].removeprefix("robotwin_").removeprefix("robotwin2_")
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            tr_seed = trial_seed[idx][0].item()
            
            wrapper = RobotwinEnvWrapper(task_name, tr_id, tr_seed, self.config, version=self.robotwin_version)
            env_wrappers.append(wrapper)
        
        # Initialize environments in parallel
        init_futures = []
        for wrapper in env_wrappers:
            future = self.env_thread_pool.submit(wrapper.initialize)
            init_futures.append(future)
        
        for future in as_completed(init_futures, timeout=360):
            try:
                future.result()
            except Exception as e:
                print(f"Environment initialization failed: {e}", flush=True)
                traceback.print_exc()
                raise
        
        # Collect initial observations
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        
        for idx, wrapper in enumerate(env_wrappers):
            try:
                obs = wrapper.get_obs()
                obs = encode_obs(obs)
                    
                task_description = wrapper.get_instruction()
                task_descriptions.append(task_description)
                inputs.append(self._obs_to_input(obs, is_robotwin=True, robotwin_version=wrapper.version))
                
                task_file_name = f"{wrapper.task_name}_trial_{wrapper.trial_id}_seed_{wrapper.trial_seed}"
                task_records.append({
                    "active": wrapper.active,
                    "complete": wrapper.complete,
                    "finish_step": wrapper.finish_step,
                    "task_file_name": task_file_name
                })
                
                if is_valid:
                    img = obs['observation']['head_camera']['rgb']
                    valid_video[task_file_name].append(img)
                    
            except Exception as e:
                print(f"Failed to get initial observation: {e}", flush=True)
                traceback.print_exc()
                raise
        
        # Main rollout loop
        step = 0
        vla_history = []
        
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
                
            current_inputs = inputs
            current_task_descriptions = task_descriptions
            
            # Get VLA actions
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            
            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "action": actions,
                "step": step
            }
            if not is_valid:
                step_data["pixel_values"] = vla_output["pixel_values"].detach().cpu().to(torch.bfloat16)
            if vla_output.get("proprio") is not None:
                step_data["proprio"] = vla_output["proprio"]

            vla_history.append(step_data)
            
            # Execute actions in parallel
            step_futures = []
            for idx in active_indices:
                future = self.env_thread_pool.submit(
                    env_wrappers[idx].step,
                    actions[idx]
                )
                step_futures.append((idx, future))
            
            # Collect results
            new_inputs = inputs.copy()
            for idx, future in step_futures:
                try:
                    obs, done = future.result(timeout=120)
                    if obs is not None:
                        obs = encode_obs(obs)
                        new_inputs[idx] = self._obs_to_input(obs, is_robotwin=True, robotwin_version=env_wrappers[idx].version)
                        
                    task_records[idx]['active'] = env_wrappers[idx].active
                    task_records[idx]['complete'] = env_wrappers[idx].complete
                    task_records[idx]['finish_step'] = env_wrappers[idx].finish_step
                    
                    if is_valid and obs is not None:
                        img = obs['observation']['head_camera']['rgb']
                        valid_video[task_records[idx]['task_file_name']].append(img)
                        
                except Exception as e:
                    print(f"Step execution failed: {e}", flush=True)
                    task_records[idx]['active'] = False
                    task_records[idx]['complete'] = False
                    task_records[idx]['finish_step'] = step + self.config.action_chunks_len
            
            inputs = new_inputs
            step += self.config.action_chunks_len
        
        # Clean up environments
        cleanup_futures = []
        for wrapper in env_wrappers:
            future = self.env_thread_pool.submit(wrapper.close)
            cleanup_futures.append(future)
            
        for future in as_completed(cleanup_futures):
            try:
                future.result(timeout=20)
            except Exception as e:
                print(f"Environment cleanup failed: {e}", flush=True)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Save validation videos
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        
        self.module.train()
        
        # Prepare output batch
        return self._prepare_output_batch(vla_history, task_records, batch_size)
    
    def _generate_minibatch_libero(self, prompts):
        """Generate minibatch for Libero using multiprocessing."""
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        task_id = prompts.batch['task_id'].repeat_interleave(n_samples, dim=0)
        trial_id = prompts.batch['trial_id'].repeat_interleave(n_samples, dim=0)
        task_suite_name = np.repeat(prompts.non_tensor_batch['task_suite_name'], n_samples)
        max_steps = self.max_steps.get(self.config.task_suite_name, 512)
        batch_size = task_id.size(0)
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        
        # Pre-load task metadata and initial states in the parent process.
        # This avoids importing torch (and libero.benchmark) inside each spawned
        # subprocess, which was costing ~5 GB per subprocess × 62 subprocesses
        # per worker = ~310 GB per worker → node OOM.
        from libero.libero import benchmark as _libero_benchmark, get_libero_path as _get_libero_path
        _benchmark_dict = _libero_benchmark.get_benchmark_dict()
        _task_suite_cache = {}
        _init_states_cache = {}
        task_bddl_files = []
        task_descriptions_pre = []
        initial_states_pre = []

        for idx in range(batch_size):
            _task_name = task_suite_name[idx]
            _t_id = task_id[idx][0].item()
            _tr_id = trial_id[idx][0].item()
            _cache_key = (_task_name, _t_id)
            if _cache_key not in _task_suite_cache:
                _task_suite_cache[_cache_key] = _benchmark_dict[_task_name]()
            _ts = _task_suite_cache[_cache_key]
            _task = _ts.get_task(_t_id)
            if _cache_key not in _init_states_cache:
                _init_states_cache[_cache_key] = _ts.get_task_init_states(_t_id)
            _initial_state = _init_states_cache[_cache_key][_tr_id]
            _bddl_file = os.path.join(
                _get_libero_path("bddl_files"), _task.problem_folder, _task.bddl_file
            )
            task_bddl_files.append(_bddl_file)
            task_descriptions_pre.append(_task.language)
            initial_states_pre.append(_initial_state)

        processes = []
        input_queues = []
        output_queues = []
        _spawn_ctx = mp_get_context('spawn')
        do_swap = getattr(self.config, 'swap_objects', False)

        # Curriculum: compute the allowed swap distance from the actual training step.
        # meta_info['global_steps'] is available even for training rollouts; the local
        # `global_steps` variable above is zeroed for training (video-logging only).
        swap_max_distance = 1e9
        if do_swap:
            _gs = meta_info.get('global_steps', 0)
            swap_max_distance = _compute_swap_max_distance(_gs, self.config)

        # Pre-compute one swap seed per unique (task, task_id, trial_id) so that all
        # n_samples rollouts in a GRPO group see the same object swap and start from
        # identical visual observations — keeping the GRPO advantage estimate unbiased.
        swap_seeds: dict = {}
        if do_swap:
            for idx in range(batch_size):
                key = (task_suite_name[idx], task_id[idx][0].item(), trial_id[idx][0].item())
                if key not in swap_seeds:
                    swap_seeds[key] = random.randint(0, 2**31 - 1)

        for idx in range(batch_size):
            task_name = task_suite_name[idx]
            t_id = task_id[idx][0].item()
            tr_id = trial_id[idx][0].item()
            swap_seed = swap_seeds.get((task_name, t_id, tr_id)) if do_swap else None
            # SAGA: look up key steps for this task's instruction
            saga_ks = self.saga_plan_configs.get(task_descriptions_pre[idx])
            input_q = _spawn_ctx.Queue()
            output_q = _spawn_ctx.Queue()
            p = _spawn_ctx.Process(
                target=_libero_env_worker,
                args=(task_bddl_files[idx], task_descriptions_pre[idx], initial_states_pre[idx],
                      task_name, t_id, tr_id, self.config, input_q, output_q, is_valid, global_steps, max_steps,
                      do_swap, swap_seed, swap_max_distance, saga_ks)
            )
            p.start()
            processes.append(p)
            input_queues.append(input_q)
            output_queues.append(output_q)
        
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = defaultdict(list)
        
        # SAGA: determine the fixed K (max substeps) for this batch
        saga_K = 2  # pick + place

        for idx in range(batch_size):
            init_data = output_queues[idx].get(timeout=120)
            assert init_data['type'] == 'init'
            task_descriptions.append(init_data["task_description"])
            inputs.append(self._obs_to_input(init_data['obs'], is_robotwin=False))
            task_records.append({
                "active": init_data['active'],
                "complete": init_data['complete'],
                "finish_step": init_data['finish_step'],
                "task_file_name": init_data['task_file_name'],
                "min_dist": init_data.get('min_dist', float('inf')),
                # SAGA: initialise with fallback values (overwritten by step results)
                "saga_substep_rewards": [0.0] * saga_K,
                "saga_substep_boundary_steps": [-1] * saga_K,
            })
            if is_valid:
                valid_video[init_data['task_file_name']].extend(init_data['valid_images'])

        step = 0
        vla_history = []

        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]

            current_inputs = inputs

            vla_input = self.process_input(current_inputs, task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            
            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "action": actions,
                "step": step
            }
            if not is_valid:
                step_data["pixel_values"] = vla_output["pixel_values"].detach().cpu().to(torch.bfloat16)
            if "proprio" in vla_output:
                step_data["proprio"] = vla_output["proprio"]
            vla_history.append(step_data)
            
            # Send actions to env workers (non-blocking)
            for idx in active_indices:
                input_queues[idx].put(actions[idx])

            # Collect env results
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = output_queues[idx].get(timeout=30)
                assert result['type'] == 'step'
                new_inputs[idx] = self._obs_to_input(result['obs'], is_robotwin=False)
                task_records[idx]['active'] = result['active']
                task_records[idx]['complete'] = result['complete']
                task_records[idx]['finish_step'] = result['finish_step']
                task_records[idx]['min_dist'] = result.get('min_dist', task_records[idx]['min_dist'])
                # SAGA: update substep tracking (latest values are cumulative)
                if 'saga_substep_rewards' in result:
                    task_records[idx]['saga_substep_rewards'] = result['saga_substep_rewards']
                    task_records[idx]['saga_substep_boundary_steps'] = result['saga_substep_boundary_steps']
                if is_valid:
                    valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])

            inputs = new_inputs
            step += self.config.action_chunks_len

        for q in input_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
        
        torch.cuda.empty_cache()
        
        if is_valid:
            for task_file, images in valid_video.items():
                complete = any(r['complete'] for r in task_records if r['task_file_name'] == task_file)
                save_rollout_video(
                    images,
                    self.config.experiment_name,
                    task_file,
                    global_steps,
                    complete
                )
        
        self.module.train()
        
        return self._prepare_output_batch(vla_history, task_records, batch_size)
    
    def _prepare_output_batch(self, vla_history, task_records, batch_size):
        """Prepare the output batch from VLA history"""
        key_names = ["responses", "input_ids", "attention_mask"]
        if "pixel_values" in vla_history[0]:
            key_names.append("pixel_values")
        if self.config.use_proprio and "proprio" in vla_history[0]:
            key_names.append("proprio")

        batch = {k: [] for k in key_names}

        for k in key_names:
            for h in vla_history:
                batch[k].append(h[k])
        
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=1)
        
        batch["complete"] = torch.tensor([bool(k["complete"]) for k in task_records], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor([k["finish_step"] for k in task_records], dtype=torch.int64, device=batch['responses'].device)
        # Cap inf at 10.0 m (effectively 0 reward) to avoid NaN in downstream kernels
        batch["min_dist"] = torch.tensor(
            [min(float(k.get("min_dist", 10.0)), 10.0) for k in task_records],
            dtype=torch.float32, device=batch['responses'].device,
        )

        # SAGA: pack substep rewards and boundary steps into batch.
        # For tasks without SAGA tracking, fallback: both substep rewards = complete
        # (degrades to standard GRPO — see saga.py docstring).
        saga_K = 2
        saga_sr = []
        saga_bs = []
        for k in task_records:
            sr = k.get("saga_substep_rewards", [0.0] * saga_K)
            bs = k.get("saga_substep_boundary_steps", [-1] * saga_K)
            # Pad/truncate to saga_K
            sr = (sr + [0.0] * saga_K)[:saga_K]
            bs = (bs + [-1] * saga_K)[:saga_K]
            # Fallback: if no SAGA data, replicate trajectory outcome for both substeps
            if all(b == -1 for b in bs):
                c = float(k.get("complete", False))
                sr = [c] * saga_K
            saga_sr.append(sr)
            saga_bs.append(bs)
        batch["saga_substep_rewards"] = torch.tensor(
            saga_sr, dtype=torch.float32, device=batch['responses'].device,
        )
        batch["saga_substep_boundary_steps"] = torch.tensor(
            saga_bs, dtype=torch.int64, device=batch['responses'].device,
        )

        output_batch = TensorDict(batch, batch_size=batch_size)
        return DataProto(batch=output_batch)
    
    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        """Generate one step of actions"""
        use_ar = getattr(self.config, 'use_autoregressive', False)
        if use_ar:
            return self._generate_one_step_autoregressive(prompts)
        elif self.config.vla == "openvla-oft":
            return self._generate_one_step_oft(prompts)
        elif self.config.vla == "openvla":
            return self._generate_one_step_openvla(prompts)
        else:
            raise ValueError(f"Unknown VLA type: {self.config.vla}")
    
    def _generate_one_step_oft(self, prompts: dict):
        """Generate one step for OpenVLA-OFT"""
        idx = prompts['input_ids']
        attention_mask = prompts['attention_mask']
        pixel_values = prompts["pixel_values"]
        proprio = prompts.get("proprio", None)
        
        param_ctx = contextlib.nullcontext()
        do_sample = prompts.get('do_sample', self.config.do_sample)
        temperature = prompts.get('temperature', self.config.temperature)
        
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                actions, response = self.module.generate_action_verl(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    proprio=proprio,
                    attention_mask=attention_mask,
                    padding_idx=self.processor.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    unnorm_key=self.config.unnorm_key,
                    temperature=temperature,
                )
        
        assert self.processor.tokenizer.pad_token_id is not None
        
        idx = verl_F.pad_sequence_to_length(
            idx,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True
        )
        
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=0,
            left_pad=True
        )
        
        batch = {
            'responses': response,
            'input_ids': idx,
            'attention_mask': attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }
        if proprio is not None:
            batch["proprio"] = proprio

        return batch

    def _generate_one_step_autoregressive(self, prompts: dict):
        """Autoregressive generation of full action chunk for OFT model loaded with use_l1_regression=False.

        Uses forward_causal_lm (standard causal-LM forward without OFT placeholder/zeroing)
        to autoregressively generate action_token_len * action_chunks_len tokens (e.g. 7*8=56).
        Returns the same batch format as _generate_one_step_oft (responses as token IDs,
        input_ids as prompt-only padded, actions as (batch, chunks, dim)).
        """
        idx = prompts['input_ids']
        attention_mask = prompts['attention_mask']
        pixel_values = prompts["pixel_values"]

        batch_size = idx.size(0)
        pad_token_id = self.processor.tokenizer.pad_token_id
        param_ctx = contextlib.nullcontext()

        do_sample = prompts.get('do_sample', self.config.do_sample)
        temperature = prompts.get('temperature', self.config.temperature)
        response_length = self.config.action_token_len * self.config.action_chunks_len  # e.g. 7*8=56

        from transformers import GenerationConfig
        generation_config = GenerationConfig(temperature=temperature)

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate_autoregressive(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )

        seq = output.sequences
        response = seq[:, idx.size(1):]

        # Pad or truncate response to exactly response_length tokens
        if response.size(1) < response_length:
            pad = torch.full(
                (batch_size, response_length - response.size(1)),
                fill_value=pad_token_id, dtype=response.dtype, device=response.device,
            )
            response = torch.cat([response, pad], dim=1)
        elif response.size(1) > response_length:
            response = response[:, :response_length]

        # Decode tokens → unnormalized actions (batch, chunks, dim)
        predicted_action_token_ids = response.detach().cpu().numpy()
        discretized_actions = self.module.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.module.bin_centers.shape[0] - 1
        )
        normalized_actions = self.module.bin_centers[discretized_actions]
        # Reshape to (batch, chunks, dims) before unnormalization so (7,) stats broadcast correctly
        normalized_actions = normalized_actions.reshape(batch_size, self.config.action_chunks_len, self.config.action_token_len)

        action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        # Return same format as _generate_one_step_oft:
        # input_ids = prompt only (padded), responses = action token IDs
        assert pad_token_id is not None
        idx = verl_F.pad_sequence_to_length(
            idx, max_seq_len=self.config.max_prompt_length,
            pad_token_id=pad_token_id, left_pad=True,
        )
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask, max_seq_len=self.config.max_prompt_length,
            pad_token_id=0, left_pad=True,
        )

        batch = {
            'responses': response,
            'input_ids': idx,
            'attention_mask': attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }
        return batch

    def _generate_one_step_openvla(self, prompts: dict):
        """Generate one step for OpenVLA — autoregressive generation of full action chunk.

        Generates action_token_len * action_chunks_len tokens (e.g. 7*8=56) autoregressively
        using the standard causal-LM forward (no OFT placeholder/zeroing), then decodes
        to unnormalized actions of shape (batch, action_chunks_len, action_token_len).
        """
        idx = prompts['input_ids']
        attention_mask = prompts['attention_mask']
        pixel_values = prompts["pixel_values"]

        eos_token_id = prompts.get('eos_token_id', None)
        pad_token_id = prompts.get('pad_token_id', self.processor.tokenizer.pad_token_id)

        batch_size = idx.size(0)
        prompt_length = idx.size(1)
        param_ctx = contextlib.nullcontext()

        do_sample = prompts.get('do_sample', self.config.do_sample)
        # Generate full action chunk: action_token_len * action_chunks_len tokens (e.g. 7*8=56)
        response_length = self.config.action_token_len * self.config.action_chunks_len
        top_p = prompts.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.get('top_k', self.config.get('top_k', 0))
        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)

        temperature = prompts.get('temperature', self.config.temperature)
        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate_autoregressive(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True
                )

        seq = output.sequences
        response = seq[:, prompt_length:]

        # Pad or truncate response to exactly response_length tokens
        if response.size(1) < response_length:
            pad = torch.full(
                (batch_size, response_length - response.size(1)),
                fill_value=pad_token_id,
                dtype=response.dtype,
                device=response.device,
            )
            response = torch.cat([response, pad], dim=1)
        elif response.size(1) > response_length:
            response = response[:, :response_length]

        seq = torch.cat([idx, response], dim=1)
        response_attention_mask = torch.ones(
            (batch_size, response_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Extract and unnormalize actions
        predicted_action_token_ids = response.detach().cpu().numpy()
        discretized_actions = self.module.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.module.bin_centers.shape[0] - 1
        )
        normalized_actions = self.module.bin_centers[discretized_actions]

        action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        # Reshape to (batch, action_chunks_len, action_token_len) e.g. (batch, 8, 7)
        actions = actions.reshape(batch_size, self.config.action_chunks_len, self.config.action_token_len)

        seq = verl_F.pad_sequence_to_length(
            seq,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True
        )
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask,
            max_seq_len=self.config.max_prompt_length,
            pad_token_id=0,
            left_pad=True
        )

        batch = {
            'responses': response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }

        return batch
    
    def _obs_to_input(self, obs, is_robotwin=False, robotwin_version="1.0"):
        """Convert observation to model input format"""
        if not is_robotwin:
            # Libero
            state = np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"]
            ])
            
            if self.config.num_images_in_input > 1:
                return {
                    "full_image": get_libero_image(obs, 224),
                    "wrist_image": get_libero_wrist_image(obs, 224),
                    "state": state
                }
            else:
                return {
                    "full_image": get_libero_image(obs, 224),
                    "state": state
                }
        else:
            # Robotwin
            if robotwin_version == "1.0":
                state = obs['joint_action']
                state[6] /= 0.045
                state[13] /= 0.045
            else:  # 2.0
                state = obs['joint_action']['vector']
            
            if self.config.num_images_in_input == 3:
                return {
                    "full_image": obs['observation']['head_camera']['rgb'],
                    "left_wrist": obs['observation']['left_camera']['rgb'],
                    "right_wrist": obs['observation']['right_camera']['rgb'],
                    "state": state
                }
            else:
                return {
                    "full_image": obs['observation']['head_camera']['rgb'],
                    "state": state
                }
    
    def __del__(self):
        """Cleanup resources on deletion"""
        if hasattr(self, 'env_thread_pool'):
            self.env_thread_pool.shutdown(wait=False)