"""
Substep-aware RL utilities for instruction-grounded RL post-training.

Provides:
- APDPlanManager: loads APD plans, provides plan lookup and wrong substep sampling
- SigCLIPRewardModel: loads SigCLIP/SigLIP, computes vision-language similarity
- SubstepTracker: per-episode substep state tracking with SigCLIP completion detection
"""

import json
import random
import logging
from typing import List, Dict, Optional
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

LIBERO_SUITE_MAPPING = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "long",
    "libero_90": "90",
}


class APDPlanManager:
    """Manages APD (Action Plan Dataset) plans for substep instruction switching
    and contrastive wrong-instruction sampling."""

    def __init__(self, apd_plans_path: str):
        with open(apd_plans_path, 'r') as f:
            self.raw_plans = json.load(f)

        self.plans_by_suite: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
        self.all_substeps_by_suite: Dict[str, List[str]] = defaultdict(list)

        for entry in self.raw_plans:
            suite = entry['suite']
            raw_instruction = entry['instruction']['raw'].lower().strip()
            plan = entry['instruction']['plan']
            self.plans_by_suite[suite][raw_instruction] = plan
            for step in plan:
                self.all_substeps_by_suite[suite].append(step['subgoal'])

        logger.info(f"[APDPlanManager] Loaded {len(self.raw_plans)} plans")
        for suite in self.plans_by_suite:
            logger.info(
                f"  {suite}: {len(self.plans_by_suite[suite])} tasks, "
                f"{len(self.all_substeps_by_suite[suite])} total substeps"
            )

    @staticmethod
    def get_suite_name(task_suite_name: str) -> str:
        return LIBERO_SUITE_MAPPING.get(task_suite_name, task_suite_name)

    def get_plan(self, task_suite_name: str, instruction: str) -> Optional[List[Dict]]:
        suite = self.get_suite_name(task_suite_name)
        key = instruction.lower().strip()
        plan = self.plans_by_suite.get(suite, {}).get(key, None)
        if plan is None:
            logger.warning(
                f"[APDPlanManager] No plan for suite={suite}, "
                f"instruction='{instruction[:60]}...'"
            )
        return plan

    def sample_wrong_substep(self, task_suite_name: str, current_subgoal: str) -> str:
        """Sample a random wrong substep instruction from the same suite."""
        suite = self.get_suite_name(task_suite_name)
        candidates = [
            s for s in self.all_substeps_by_suite.get(suite, [])
            if s != current_subgoal
        ]
        if not candidates:
            return "do nothing"
        return random.choice(candidates)


class SigCLIPRewardModel:
    """Lightweight SigCLIP/SigLIP wrapper for vision-language similarity."""

    def __init__(self, model_path: str = "timm/ViT-B-16-SigLIP-256",
                 device: torch.device = None):
        self.device = device or torch.device('cuda')
        self.is_openclip = model_path.startswith("timm/")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        if self.is_openclip:
            from open_clip import create_model_from_pretrained, get_tokenizer
            name = f"hf-hub:{model_path}"
            logger.info(f"[SigCLIP] Loading open_clip: {name}")
            self.model, self.preprocess = create_model_from_pretrained(name)
            self.tokenizer = get_tokenizer(name)
        else:
            from transformers import AutoModel, AutoProcessor
            logger.info(f"[SigCLIP] Loading transformers: {model_path}")
            self.model = AutoModel.from_pretrained(model_path)
            self.preprocess = AutoProcessor.from_pretrained(model_path)
        self.model = self.model.to(self.device).eval()
        logger.info("[SigCLIP] Model loaded successfully")

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into normalized embeddings. Shape: (N, D)."""
        if self.is_openclip:
            tokens = self.tokenizer(
                texts, context_length=self.model.context_length
            ).to(self.device)
            return self.model.encode_text(tokens, normalize=True)
        else:
            inputs = self.preprocess(
                text=texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            embeds = self.model.get_text_features(**inputs)
            return embeds / embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode an RGB uint8 numpy image into a normalized embedding. Shape: (1, D)."""
        pil = Image.fromarray(image)
        if self.is_openclip:
            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
            return self.model.encode_image(tensor, normalize=True)
        else:
            inputs = self.preprocess(images=pil, return_tensors="pt").to(self.device)
            embeds = self.model.get_image_features(**inputs)
            return embeds / embeds.norm(dim=-1, keepdim=True)

    def compute_similarity(self, image: np.ndarray,
                           text_embedding: torch.Tensor) -> float:
        """Cosine similarity between an image and a precomputed text embedding."""
        img_embed = self.encode_image(image)
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        return float(torch.cosine_similarity(img_embed, text_embedding, dim=-1).item())


class SubstepTracker:
    """Tracks substep progress for a single episode during RL rollout."""

    def __init__(self, plan: List[Dict], sigclip: SigCLIPRewardModel,
                 threshold: float = 0.25):
        self.plan = plan
        self.sigclip = sigclip
        self.threshold = threshold
        self.current_idx = 0

        effects = [step['expected_effect'] for step in plan]
        self.text_embeddings = sigclip.encode_texts(effects)

    def get_current_instruction(self) -> str:
        if self.current_idx >= len(self.plan):
            return self.plan[-1]['subgoal']
        return self.plan[self.current_idx]['subgoal']

    def check_and_advance(self, image: np.ndarray) -> bool:
        """Check if current substep is complete; advance if so. Returns True if advanced."""
        if self.current_idx >= len(self.plan) - 1:
            return False
        sim = self.sigclip.compute_similarity(
            image, self.text_embeddings[self.current_idx]
        )
        if sim >= self.threshold:
            self.current_idx += 1
            return True
        return False
