"""
InfoBot-VLA: Information Bottleneck Constrained VLA

Core innovation: Force VLA to predict actions through a language bottleneck,
preventing direct visual overfitting and addressing H(L|V) ≈ 0 problem.

Mathematical formulation:
    L_total = L_action + β * I(Z_v; V | L)
    
Where:
    - I(Z_v; V | L) is the conditional mutual information (minimize)
    - This encourages Z_v to only contain information relevant to L
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class InfoBottleneckLayer(nn.Module):
    """
    Information Bottleneck Layer: Compress visual features conditioned on language.
    
    Key insight: Visual information must pass through a narrow bottleneck
    that is conditioned on language. This forces the model to only extract
    visual information relevant to the language instruction.
    
    Architecture:
        V_features -> CrossAttention(Q=language, K=V_features, V=V_features) 
                   -> MLP compression -> Z_v (bottleneck representation)
    """
    def __init__(
        self,
        visual_dim: int,
        language_dim: int,
        bottleneck_dim: int = 256,  # Compressed dimension
        num_bottleneck_tokens: int = 8,  # Number of bottleneck tokens
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        
        # Cross-attention: Language queries visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=language_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Project visual features to language dimension for attention
        self.visual_to_lang = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, language_dim)
        )
        
        # Learnable bottleneck tokens (initialized with smaller variance)
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottleneck_tokens, language_dim) * 0.02
        )
        
        # MLP for further compression
        self.compression_mlp = nn.Sequential(
            nn.LayerNorm(language_dim),
            nn.Linear(language_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(language_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, N_v, D_v)
        language_features: torch.Tensor,  # (B, N_l, D_l)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            visual_features: Visual patch features from vision backbone
            language_features: Language features from LLM embedding
            attention_mask: Optional mask for visual features
            
        Returns:
            bottleneck_features: (B, num_bottleneck_tokens, bottleneck_dim)
            info_dict: Dictionary with auxiliary outputs for loss computation
        """
        batch_size = visual_features.shape[0]
        
        # Project visual features to language dimension
        visual_proj = self.visual_to_lang(visual_features)  # (B, N_v, D_l)
        
        # Expand bottleneck tokens for batch
        bottleneck_queries = self.bottleneck_tokens.expand(batch_size, -1, -1)  # (B, K, D_l)
        
        # Cross-attention: Bottleneck tokens attend to visual features
        # This extracts visual information most relevant to the bottleneck
        attn_output, attn_weights = self.cross_attn(
            query=bottleneck_queries,
            key=visual_proj,
            value=visual_proj,
            key_padding_mask=attention_mask,
            need_weights=True,
        )  # attn_output: (B, K, D_l), attn_weights: (B, K, N_v)
        
        # Layer norm and residual
        attn_output = self.norm(attn_output + bottleneck_queries)
        
        # Compress to bottleneck dimension
        bottleneck_features = self.compression_mlp(attn_output)  # (B, K, D_b)
        
        info_dict = {
            'attn_weights': attn_weights,
            'pre_compression': attn_output,
        }
        
        return bottleneck_features, info_dict


class LanguageConditionedBottleneck(nn.Module):
    """
    Language-Conditioned Bottleneck: V2 with stronger language conditioning
    
    This version explicitly models p(Z_v | L) by using language features
    to parameterize the bottleneck compression.
    """
    def __init__(
        self,
        visual_dim: int,
        language_dim: int,
        bottleneck_dim: int = 256,
        num_bottleneck_tokens: int = 8,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        
        # Language-to-bottleneck projection
        # This creates language-conditioned bottleneck parameters
        self.lang_to_bottleneck = nn.Sequential(
            nn.Linear(language_dim, language_dim),
            nn.LayerNorm(language_dim),
            nn.GELU(),
            nn.Linear(language_dim, num_bottleneck_tokens * bottleneck_dim),
        )
        
        # Visual projection
        self.visual_proj = nn.Linear(visual_dim, bottleneck_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,  # (B, N_v, D_v)
        language_features: torch.Tensor,  # (B, N_l, D_l)
    ) -> torch.Tensor:
        """
        Args:
            visual_features: Visual features
            language_features: Language features (use [CLS] or mean pooling)
            
        Returns:
            bottleneck_features: (B, num_bottleneck_tokens, bottleneck_dim)
        """
        batch_size = visual_features.shape[0]
        
        # Aggregate language features (mean pooling over tokens)
        lang_agg = language_features.mean(dim=1)  # (B, D_l)
        
        # Generate language-conditioned bottleneck parameters
        lang_conditioned = self.lang_to_bottleneck(lang_agg)  # (B, K * D_b)
        lang_conditioned = lang_conditioned.view(
            batch_size, self.num_bottleneck_tokens, self.bottleneck_dim
        )  # (B, K, D_b)
        
        # Project and pool visual features
        visual_proj = self.visual_proj(visual_features)  # (B, N_v, D_b)
        # Pool visual features to match bottleneck tokens
        visual_pooled = visual_proj.mean(dim=1, keepdim=True)  # (B, 1, D_b)
        visual_pooled = visual_pooled.expand(-1, self.num_bottleneck_tokens, -1)  # (B, K, D_b)
        
        # Fuse language-conditioned and visual features
        fused = lang_conditioned + visual_pooled  # (B, K, D_b)
        bottleneck_features = self.fusion(fused)  # (B, K, D_b)
        
        return bottleneck_features


class MutualInformationEstimator(nn.Module):
    """
    Mutual Information Estimator using InfoNCE-based approximation.
    
    Estimates I(Z_v; V | L) using contrastive learning.
    """
    def __init__(self, bottleneck_dim: int, visual_dim: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
        # Projections for MI estimation
        self.bottleneck_proj = nn.Linear(bottleneck_dim, 128)
        self.visual_proj = nn.Linear(visual_dim, 128)
        
    def forward(
        self,
        bottleneck_features: torch.Tensor,  # (B, K, D_b)
        visual_features: torch.Tensor,  # (B, N_v, D_v)
    ) -> torch.Tensor:
        """
        Estimate mutual information using InfoNCE lower bound.
        
        Args:
            bottleneck_features: Bottleneck representations
            visual_features: Original visual features
            
        Returns:
            mi_estimate: Scalar MI estimate
        """
        # Pool bottleneck features
        z_v = bottleneck_features.mean(dim=1)  # (B, D_b)
        
        # Pool visual features
        v = visual_features.mean(dim=1)  # (B, D_v)
        
        # Project to common space
        z_v_proj = F.normalize(self.bottleneck_proj(z_v), dim=-1)  # (B, 128)
        v_proj = F.normalize(self.visual_proj(v), dim=-1)  # (B, 128)
        
        # Clamp projections to avoid extreme values
        z_v_proj = torch.clamp(z_v_proj, min=-10, max=10)
        v_proj = torch.clamp(v_proj, min=-10, max=10)
        
        # Compute similarity matrix with numerical stability
        logits = torch.matmul(z_v_proj, v_proj.T) / self.temperature  # (B, B)
        
        # Clamp logits to prevent overflow in softmax
        logits = torch.clamp(logits, min=-50, max=50)
        
        # InfoNCE loss: positive pairs on diagonal
        labels = torch.arange(len(logits), device=logits.device)
        
        # Symmetric InfoNCE
        loss_i2v = F.cross_entropy(logits, labels)
        loss_v2i = F.cross_entropy(logits.T, labels)
        
        # Check for NaN/Inf
        if not torch.isfinite(loss_i2v) or not torch.isfinite(loss_v2i):
            # Return a small constant loss instead of NaN
            return torch.tensor(1.0, device=logits.device)
        
        mi_estimate = (loss_i2v + loss_v2i) / 2
        
        return mi_estimate


class InfoBotActionHead(nn.Module):
    """
    Action head that takes bottleneck features and language features,
    ensuring action prediction depends on both.
    """
    def __init__(
        self,
        bottleneck_dim: int,
        language_dim: int,
        hidden_dim: int,
        action_dim: int = 7,
        num_action_chunks: int = 8,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.language_dim = language_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        
        # Cross-attention: Action tokens attend to bottleneck + language
        combined_dim = bottleneck_dim + language_dim
        
        self.action_queries = nn.Parameter(
            torch.randn(1, num_action_chunks * action_dim, hidden_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # Project bottleneck+language to query dimension
        self.context_proj = nn.Linear(combined_dim, hidden_dim)
        
        # MLP for action prediction
        self.action_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # Predict one action dimension per query
        )
        
    def forward(
        self,
        bottleneck_features: torch.Tensor,  # (B, K, D_b)
        language_features: torch.Tensor,  # (B, N_l, D_l)
    ) -> torch.Tensor:
        """
        Predict actions from bottleneck and language features.
        
        Args:
            bottleneck_features: Bottleneck visual representations
            language_features: Language representations
            
        Returns:
            actions: (B, num_action_chunks, action_dim)
        """
        batch_size = bottleneck_features.shape[0]
        
        # Pool language features
        lang_pooled = language_features.mean(dim=1, keepdim=True)  # (B, 1, D_l)
        
        # Expand to match bottleneck tokens
        lang_expanded = lang_pooled.expand(-1, bottleneck_features.shape[1], -1)  # (B, K, D_l)
        
        # Concatenate bottleneck and language
        context = torch.cat([bottleneck_features, lang_expanded], dim=-1)  # (B, K, D_b + D_l)
        context = self.context_proj(context)  # (B, K, D_h)
        
        # Expand action queries
        action_queries = self.action_queries.expand(batch_size, -1, -1)  # (B, N_a, D_h)
        
        # Cross-attention: Action queries attend to context
        attn_output, _ = self.cross_attn(
            query=action_queries,
            key=context,
            value=context,
        )  # (B, N_a, D_h)
        
        # Predict actions
        action_logits = self.action_mlp(attn_output)  # (B, N_a, 1)
        action_logits = action_logits.squeeze(-1)  # (B, N_a)
        
        # Reshape to (B, num_chunks, action_dim)
        actions = action_logits.view(batch_size, self.num_action_chunks, self.action_dim)
        
        return actions


def compute_infobot_loss(
    action_pred: torch.Tensor,
    action_gt: torch.Tensor,
    bottleneck_features: torch.Tensor,
    visual_features: torch.Tensor,
    language_features: torch.Tensor,
    mi_estimator: MutualInformationEstimator,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute InfoBot-VLA total loss.
    
    L_total = L_action + β * I(Z_v; V | L)
    
    Args:
        action_pred: Predicted actions
        action_gt: Ground truth actions
        bottleneck_features: Bottleneck representations
        visual_features: Original visual features
        language_features: Language features
        mi_estimator: Mutual information estimator module
        beta: Weight for MI regularization
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    # Action prediction loss
    action_loss = F.l1_loss(action_pred, action_gt)
    
    # Mutual information regularization
    # We want to minimize I(Z_v; V | L), which encourages Z_v to only contain
    # information relevant to the task (via language)
    mi_loss = mi_estimator(bottleneck_features, visual_features)
    
    # Total loss
    total_loss = action_loss + beta * mi_loss
    
    loss_dict = {
        'action_loss': action_loss.item(),
        'mi_loss': mi_loss.item(),
        'total_loss': total_loss.item(),
        'beta': beta,
    }
    
    return total_loss, loss_dict


class InfoBotVLAModel(nn.Module):
    """
    InfoBot-VLA Model wrapper that adds information bottleneck to OpenVLA.
    
    Architecture:
        Vision Backbone → Projector → InfoBottleneck (conditioned on Language) → Action Head
    """
    def __init__(
        self,
        base_vla,
        bottleneck_type: str = "cross_attn",
        bottleneck_dim: int = 256,
        num_bottleneck_tokens: int = 8,
        use_l1_regression: bool = True,
    ):
        super().__init__()
        self.base_vla = base_vla
        self.llm_dim = base_vla.llm_dim
        self.bottleneck_type = bottleneck_type
        self.bottleneck_dim = bottleneck_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        
        # Vision backbone and projector from base VLA
        self.vision_backbone = base_vla.vision_backbone
        self.projector = base_vla.projector
        
        # Language model from base VLA (for getting language embeddings)
        self.language_model = base_vla.language_model
        
        # InfoBot Bottleneck
        if bottleneck_type == "cross_attn":
            self.bottleneck = InfoBottleneckLayer(
                visual_dim=self.llm_dim,
                language_dim=self.llm_dim,
                bottleneck_dim=bottleneck_dim,
                num_bottleneck_tokens=num_bottleneck_tokens,
            )
        elif bottleneck_type == "lang_conditioned":
            self.bottleneck = LanguageConditionedBottleneck(
                visual_dim=self.llm_dim,
                language_dim=self.llm_dim,
                bottleneck_dim=bottleneck_dim,
                num_bottleneck_tokens=num_bottleneck_tokens,
            )
        else:
            raise ValueError(f"Unknown bottleneck type: {bottleneck_type}")
        
        # Action head (replaces base VLA's action prediction)
        if use_l1_regression:
            from prismatic.models.action_heads import L1RegressionActionHead
            from prismatic.vla.constants import ACTION_DIM
            self.action_head = L1RegressionActionHead(
                input_dim=bottleneck_dim,
                hidden_dim=self.llm_dim,
                action_dim=ACTION_DIM,
            )
        
        # Flag for returning bottleneck features during training
        self.return_bottleneck_features = False

    @property
    def norm_stats(self):
        """Delegate norm_stats to base_vla so external code can access normalization statistics."""
        return self.base_vla.norm_stats

    @norm_stats.setter
    def norm_stats(self, value):
        self.base_vla.norm_stats = value

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
    ):
        """
        Forward pass with InfoBot bottleneck.
        """
        # Get input embeddings from language model
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)
        
        # Get visual features
        patch_features = self.vision_backbone(pixel_values)
        projected_visual = self.projector(patch_features)
        
        # Add proprio if provided
        if proprio_projector is not None and proprio is not None:
            proprio = proprio.reshape(projected_visual.shape[0], -1)
            proprio_features = proprio_projector(proprio).unsqueeze(1)
            projected_visual = torch.cat([projected_visual, proprio_features], dim=1)
        
        # Use input_embeddings as language features (simplified)
        lang_embeddings = input_embeddings
        
        # Apply InfoBot bottleneck
        if self.bottleneck_type == "cross_attn":
            bottleneck_features, bottleneck_info = self.bottleneck(
                visual_features=projected_visual,
                language_features=lang_embeddings,
            )
        else:
            bottleneck_features = self.bottleneck(
                visual_features=projected_visual,
                language_features=lang_embeddings,
            )
            bottleneck_info = {}
        
        # Get action hidden states from bottleneck
        batch_size = bottleneck_features.shape[0]
        from prismatic.vla.constants import ACTION_DIM
        actions_hidden_states = bottleneck_features.unsqueeze(2).expand(
            -1, -1, ACTION_DIM, -1
        ).reshape(batch_size, self.num_bottleneck_tokens * ACTION_DIM, self.bottleneck_dim)
        
        # Predict actions
        predicted_actions = self.action_head.predict_action(actions_hidden_states)
        
        output = {
            'predicted_actions': predicted_actions,
            'bottleneck_features': bottleneck_features,
            'visual_features': projected_visual,
            'language_embeddings': lang_embeddings,
        }
        
        if self.return_bottleneck_features or self.training:
            output['bottleneck_info'] = bottleneck_info
        
        return output
    
    def predict_action(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unnorm_key: str,
        do_sample: bool = False,
        **kwargs
    ):
        """
        Predict actions for evaluation.
        
        This method matches the interface of OpenVLAForActionPrediction.predict_action()
        """
        batch_size = pixel_values.shape[0]
        
        # Get proprio if provided
        proprio = kwargs.get('proprio', None)
        proprio_projector = kwargs.get('proprio_projector', None)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                proprio=proprio,
                proprio_projector=proprio_projector,
            )
        
        predicted_actions = output['predicted_actions']
        
        # Unnormalize actions
        if unnorm_key in self.base_vla.norm_stats:
            action_norm_stats = self.base_vla.norm_stats[unnorm_key]["action"]
            action_mean = torch.tensor(action_norm_stats["mean"], device=predicted_actions.device, dtype=predicted_actions.dtype)
            action_std = torch.tensor(action_norm_stats["std"], device=predicted_actions.device, dtype=predicted_actions.dtype)
            predicted_actions = predicted_actions * action_std + action_mean
        
        # Convert to list of numpy arrays
        actions = []
        for i in range(predicted_actions.shape[0]):
            actions.append(predicted_actions[i].cpu().float().numpy())
        
        # Return format compatible with get_vla_action
        if len(actions) == 1:
            return actions[0], None, None, None
        return actions, None, None, None
