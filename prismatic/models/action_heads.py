"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=None,  # Will use ACTION_DIM from constants if not specified
    ):
        super().__init__()
        # Use ACTION_DIM from constants if action_dim not specified
        if action_dim is None:
            action_dim = ACTION_DIM
        self.action_dim = action_dim
        
        self.model = MLPResNet(
            num_blocks=2, 
            input_dim=input_dim*action_dim,
            hidden_dim=hidden_dim, 
            output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, NUM_ACTIONS_CHUNK * action_dim, hidden_dim)
        # Output:
        # - shape: (batch_size, NUM_ACTIONS_CHUNK, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        # Reshape: (B, NUM_ACTIONS_CHUNK * action_dim, hidden_dim) 
        #       -> (B, NUM_ACTIONS_CHUNK, action_dim * hidden_dim)
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=None,  # Will use ACTION_DIM from constants if not specified
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        # Use ACTION_DIM from constants if action_dim not specified
        if action_dim is None:
            from prismatic.vla.constants import ACTION_DIM
            action_dim = ACTION_DIM
        self.action_dim = action_dim
        
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*action_dim,
            hidden_dim=hidden_dim, 
            action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred


class EOSClassificationHead(nn.Module):
    """
    独立的 EOS 分类头，用于预测 substep 结束位置。
    
    设计要点：
    - 输入：action hidden states (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D)
    - 输出：EOS logits (B, NUM_ACTIONS_CHUNK, 1) - 每个action位置一个logit
    - 结构：2层 MLP + BatchNorm + Dropout (防止过拟合)
    - 激活：最后不加sigmoid（留给BCEWithLogitsLoss）
    
    与 L1RegressionActionHead 的区别：
    - 用途不同：分类 vs 回归
    - 输出维度：1 (logit) vs ACTION_DIM (7)
    - 损失函数：BCE vs L1
    """
    def __init__(
        self,
        input_dim=4096,      # LLM hidden dim
        hidden_dim=1024,     # MLP hidden dim (可以比action head小)
        dropout=0.1,         # Dropout率
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 计算输入特征维度：input_dim * ACTION_DIM
        # 因为每个action的hidden states是 ACTION_DIM 个token的拼接
        from prismatic.vla.constants import ACTION_DIM
        feature_dim = input_dim * ACTION_DIM
        
        # 2层MLP结构
        self.model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # 输出单个logit
        )
    
    def forward(self, actions_hidden_states):
        """
        预测每个action位置的EOS logit
        
        Args:
            actions_hidden_states: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
                                   每个action由ACTION_DIM个token组成
        
        Returns:
            eos_logits: (B, NUM_ACTIONS_CHUNK, 1)
                       每个action位置一个logit值
        """
        batch_size = actions_hidden_states.shape[0]
        # Reshape: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, D) 
        #       -> (B, NUM_ACTIONS_CHUNK, ACTION_DIM * D)
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        # MLP: (B, NUM_ACTIONS_CHUNK, ACTION_DIM * D) -> (B, NUM_ACTIONS_CHUNK, 1)
        eos_logits = self.model(rearranged)
        return eos_logits


class EOSBufferManager:
    """
    EOS 样本缓冲区管理器 - 优化版本（修复梯度问题）
    
    核心改进：
    1. 存储hidden states而不是logits，保留梯度计算能力
    2. 固定容量buffer (如 50个EOS=0 + 50个EOS=1)
    3. 每次随机采样固定数量 (如 20个EOS=0 + 15个EOS=1)
    4. 采样时重新通过eos_head forward得到有梯度的logits
    5. 使用deque实现FIFO，自动丢弃最旧样本
    
    优势：
    - 正确保留梯度，EOS head能被训练
    - 更高的训练效率（更频繁的更新）
    - 更好的样本多样性（随机采样 + 滚动buffer）
    - 稳定的正负样本比例
    """
    def __init__(
        self,
        buffer_capacity=50,     # 每类样本的buffer容量
        sample_positive=15,     # 每次采样的正样本数
        sample_negative=20,     # 每次采样的负样本数
        min_positive=15,        # 触发更新的最小正样本数
        min_negative=20,        # 触发更新的最小负样本数
        device='cuda'
    ):
        from collections import deque
        
        self.buffer_capacity = buffer_capacity
        self.sample_positive = sample_positive
        self.sample_negative = sample_negative
        self.min_positive = min_positive
        self.min_negative = min_negative
        self.device = device
        
        # 使用deque实现固定容量的FIFO buffer
        # 存储单个样本的 (hidden_state, label) 元组
        # hidden_state是detached的，但在采样时会重新forward得到有梯度的logits
        self.positive_buffer = deque(maxlen=buffer_capacity)
        self.negative_buffer = deque(maxlen=buffer_capacity)
        
        # 统计信息
        self.total_positive_seen = 0
        self.total_negative_seen = 0
        self.total_updates = 0
    
    def add_samples(self, hidden_states, eos_labels):
        """
        添加新样本到缓冲区
        
        Args:
            hidden_states: (N, ACTION_DIM, hidden_dim) tensor - 每个样本包含ACTION_DIM个hidden states
            eos_labels: (N,) tensor - EOS真实标签 (0 or 1)
        """
        # Detach hidden states以避免保留计算图，但保存它们以便后续重新forward
        hidden_states = hidden_states.detach().to(self.device)
        eos_labels = eos_labels.detach().to(self.device)
        
        # 分离正负样本
        positive_mask = eos_labels > 0.5
        negative_mask = eos_labels <= 0.5
        
        # 逐个添加到对应缓冲区 (deque会自动处理容量限制)
        if positive_mask.any():
            pos_hidden = hidden_states[positive_mask]  # (num_pos, ACTION_DIM, hidden_dim)
            pos_labels = eos_labels[positive_mask]
            for hidden, label in zip(pos_hidden, pos_labels):
                # hidden: (ACTION_DIM, hidden_dim)
                self.positive_buffer.append((hidden, label))
            self.total_positive_seen += len(pos_hidden)
        
        if negative_mask.any():
            neg_hidden = hidden_states[negative_mask]  # (num_neg, ACTION_DIM, hidden_dim)
            neg_labels = eos_labels[negative_mask]
            for hidden, label in zip(neg_hidden, neg_labels):
                # hidden: (ACTION_DIM, hidden_dim)
                self.negative_buffer.append((hidden, label))
            self.total_negative_seen += len(neg_hidden)
    
    def can_compute_loss(self):
        """检查是否累积了足够的样本"""
        return (len(self.positive_buffer) >= self.min_positive and 
                len(self.negative_buffer) >= self.min_negative)
    
    def get_balanced_batch(self, eos_head):
        """
        随机采样平衡的batch，重新forward得到有梯度的logits
        
        Args:
            eos_head: EOS classification head用于重新计算logits
        
        Returns:
            balanced_logits: (sample_positive + sample_negative,) tensor - 有梯度
            balanced_labels: (sample_positive + sample_negative,) tensor
        """
        import random
        
        # 从buffer中随机采样（不移除样本）
        pos_samples = random.sample(list(self.positive_buffer), self.sample_positive)
        neg_samples = random.sample(list(self.negative_buffer), self.sample_negative)
        
        # 解包样本
        pos_hidden, pos_labels = zip(*pos_samples)
        neg_hidden, neg_labels = zip(*neg_samples)
        
        # 转换为tensor
        # pos_hidden/neg_hidden是list of (ACTION_DIM, hidden_dim)
        pos_hidden = torch.stack(list(pos_hidden))  # (sample_positive, ACTION_DIM, hidden_dim)
        pos_labels = torch.stack(list(pos_labels))
        neg_hidden = torch.stack(list(neg_hidden))  # (sample_negative, ACTION_DIM, hidden_dim)
        neg_labels = torch.stack(list(neg_labels))
        
        # 合并正负样本的hidden states
        balanced_hidden = torch.cat([pos_hidden, neg_hidden])  # (total_samples, ACTION_DIM, hidden_dim)
        balanced_labels = torch.cat([pos_labels, neg_labels])
        
        # 随机打乱
        shuffle_indices = torch.randperm(len(balanced_hidden), device=self.device)
        balanced_hidden = balanced_hidden[shuffle_indices]
        balanced_labels = balanced_labels[shuffle_indices]
        
        # 重新forward得到有梯度的logits
        # EOSClassificationHead期望输入: (B, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
        # 但我们这里每个样本只有一个action chunk，所以需要reshape为: (B, ACTION_DIM, hidden_dim)
        # 然后flatten: (B, ACTION_DIM * hidden_dim)
        batch_size = balanced_hidden.shape[0]
        hidden_dim = balanced_hidden.shape[2]
        
        # 导入ACTION_DIM常量
        from prismatic.vla.constants import ACTION_DIM as ACT_DIM
        
        # Reshape为eos_head期望的格式
        # (batch_size, ACTION_DIM, hidden_dim) -> (batch_size, 1, ACTION_DIM * hidden_dim)
        balanced_hidden_reshaped = balanced_hidden.reshape(batch_size, 1, ACT_DIM * hidden_dim)
        
        # Forward通过eos_head（保留梯度）
        # 需要先将 (batch_size, 1, ACTION_DIM * hidden_dim) reshape为 
        # (batch_size, 1 * ACTION_DIM, hidden_dim) 以匹配eos_head的输入期望
        # 但这样会有问题，因为eos_head期望 NUM_ACTIONS_CHUNK * ACTION_DIM 个tokens
        
        # 更简单的方法：直接调用eos_head的内部model（跳过reshape）
        # 因为我们已经有了正确的feature格式
        with torch.enable_grad():
            # 直接使用MLP处理 (batch_size, 1, ACTION_DIM * hidden_dim)
            balanced_logits = eos_head.model(balanced_hidden_reshaped)  # (batch_size, 1, 1)
            balanced_logits = balanced_logits.squeeze(-1).squeeze(-1)  # (batch_size,)
        
        self.total_updates += 1
        
        return balanced_logits, balanced_labels
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'buffer_positive': len(self.positive_buffer),
            'buffer_negative': len(self.negative_buffer),
            'total_positive_seen': self.total_positive_seen,
            'total_negative_seen': self.total_negative_seen,
            'total_updates': self.total_updates,
            'positive_utilization': self.total_updates * self.sample_positive / max(1, self.total_positive_seen),
            'negative_utilization': self.total_updates * self.sample_negative / max(1, self.total_negative_seen),
        }
    
    def reset(self):
        """重置缓冲区"""
        self.positive_buffer.clear()
        self.negative_buffer.clear()
        self.total_positive_seen = 0
        self.total_negative_seen = 0
        self.total_updates = 0


def compute_classification_metrics(pred, gt):
    """
    计算二分类指标
    
    Args:
        pred: (N,) bool tensor - 预测结果 (True/False)
        gt: (N,) bool tensor - 真实标签 (True/False)
    
    Returns:
        dict: 包含 accuracy, precision, recall, F1 和混淆矩阵的指标字典
    """
    import torch
    
    # 确保是 bool tensor
    pred = pred.bool()
    gt = gt.bool()
    
    # 计算混淆矩阵元素
    # True Positives, False Positives, True Negatives, False Negatives
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    tn = (~pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    
    # 计算指标
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "eos_accuracy": accuracy.item(),
        "eos_precision": precision.item(),
        "eos_recall": recall.item(),
        "eos_f1": f1.item(),
        "eos_tp": int(tp.item()),
        "eos_fp": int(fp.item()),
        "eos_tn": int(tn.item()),
        "eos_fn": int(fn.item()),
    }
