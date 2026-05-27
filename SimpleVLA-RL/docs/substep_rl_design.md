# Instruction-Grounded RL (IG-RL) Post-Training 设计文档

## 1. 项目背景与动机

### 1.1 问题：指令忽视 (Instruction Ignoring)

Vision-Language-Action (VLA) 模型在 SFT 训练后，倾向于过度依赖视觉线索，而对语言指令不敏感。即使给出不同的自然语言指令，模型在相似视觉场景下会产出近乎相同的动作序列。这在需要根据不同指令执行不同操作的场景（如 LIBERO 多任务环境）中表现尤为突出。

### 1.2 已有工作

- **openvla-oft substep SFT**：通过 Action Plan Dataset (APD) 将高层任务指令分解为细粒度子步骤指令（subgoals），在 SFT 阶段让模型学习跟随更具体的子步骤描述，增强语言接地能力。
- **SimpleVLA-RL**：基于 GRPO（Group Relative Policy Optimization）的 RL 后训练框架，使用稀疏二值任务奖励（成功=1，失败=0）进行策略优化。

### 1.3 本方案目标

在 substep SFT checkpoint 基础上，通过 **RL 后训练** 进一步强化模型的指令敏感性。核心思路是在 RL rollout 中引入 **对比奖励 (Contrastive Reward)**，显式奖励模型对不同指令产出不同动作的能力。

---

## 2. 方案设计

### 2.1 总体架构

```
substep SFT checkpoint
        │
        ▼
┌─────────────────────────────────────┐
│  IG-RL Post-Training (GRPO)         │
│                                     │
│  Rollout 阶段:                       │
│  ┌─────────────────────────────┐    │
│  │ 1. APD 子步骤指令动态切换     │    │
│  │ 2. SigCLIP 子步骤完成检测     │    │
│  │ 3. 对比 forward pass 采样    │    │
│  └─────────────────────────────┘    │
│                                     │
│  Reward 函数:                        │
│  R = w1 * R_task + w2 * R_contrastive│
│      (5)           (2)              │
│                                     │
│  优势估计: GRPO (不变)               │
└─────────────────────────────────────┘
```

### 2.2 Reward 函数

$$R_{total} = w_1 \cdot R_{task} + w_2 \cdot R_{contrastive}$$

| 组件 | 权重 | 类型 | 说明 |
|------|------|------|------|
| $R_{task}$ | $w_1 = 5$ | 稀疏二值 | 任务成功=1，失败=0，仅在 episode 结束给出 |
| $R_{contrastive}$ | $w_2 = 2$ | 半稠密标量 | 衡量模型对正确/错误指令的动作区分度，范围 [0, 1] |

### 2.3 R_contrastive 计算流程

在 rollout 过程中，每隔 K=16 个环境步（约每 2 个 VLA 推理 step）：

1. **正常 forward pass**: 用当前 substep 指令 + 当前观测 → 正确动作 $a_{correct}$
2. **对比 forward pass**: 用**同 suite 其他任务**的随机 substep 指令 + 相同观测 → 错误动作 $a_{wrong}$
3. **计算行为差异**:

$$\text{score} = \min\left(\frac{\|a_{correct} - a_{wrong}\|_2}{\sqrt{d}}, \; 1.0\right)$$

其中 $d$ 是动作维度数。score ∈ [0, 1]，越大表示模型越区分不同指令。

4. 整个 episode 的 R_contrastive 取所有采样点的**算术平均值**。

**直觉**：如果模型真的在"听"指令，给正确和错误指令应该产出不同动作（score 高）；如果模型忽视指令只看图像，两个动作几乎一样（score 趋近 0）。

### 2.4 子步骤切换机制

Rollout 时不再使用固定的高层任务指令，而是动态使用 APD 定义的子步骤 subgoal：

1. **APDPlanManager** 加载 `APD_plans.json`，提供计划查询和错误 substep 采样
2. **SubstepTracker** 跟踪每个 episode 的子步骤进度
3. **SigCLIPRewardModel** 判断子步骤是否完成：将当前画面与子步骤的 `expected_effect` 计算余弦相似度，超过阈值（0.25）则前进到下一子步骤

### 2.5 具体 Reward 示例

假设一个 episode 跑了 50 步，K=16，任务成功：

| 环境步 | 采样 | normalized_diff |
|--------|------|-----------------|
| 0 | 跳过 (step=0) | - |
| 16 | 是 | 0.35 |
| 32 | 是 | 0.42 |
| 48 | 是 | 0.28 |

- `avg_contrastive = (0.35 + 0.42 + 0.28) / 3 = 0.35`
- 任务成功: $R = 5 \times 1.0 + 2 \times 0.35 = 5.7$
- 任务失败: $R = 5 \times 0.0 + 2 \times 0.35 = 0.7$

关键: 即使任务失败，有指令敏感性的 trajectory 也能获得正向 reward（0.7），GRPO 可以区分"失败但听指令"和"失败且忽视指令"的 trajectory。

---

## 3. 代码修改详情

所有修改均在 `SimpleVLA-RL/` 目录下，通过配置开关 `use_substep_rl` 控制，**完全向后兼容**原始 SimpleVLA-RL 功能。

### 3.1 新增文件

#### `verl/utils/substep_reward.py`

封装子步骤 RL 的核心工具类：

- **`APDPlanManager`**: 加载 `APD_plans.json`，提供:
  - `get_plan(suite, instruction)`: 查询任务对应的子步骤计划
  - `sample_wrong_substep(suite, current_subgoal)`: 从同 suite 其他任务中随机采样一个"错误"substep 指令（用于对比 forward pass）
  - `LIBERO_SUITE_MAPPING`: 标准化 suite 名称映射 (如 `libero_10` → `long`)

- **`SigCLIPRewardModel`**: 轻量 SigLIP/open_clip 封装:
  - 复用 openvla-oft 中已验证的 `timm/ViT-B-16-SigLIP-256`
  - `encode_texts()` / `encode_image()`: 生成归一化嵌入
  - `compute_similarity()`: 图像-文本余弦相似度

- **`SubstepTracker`**: 每个 episode 的子步骤状态机:
  - 预计算所有 `expected_effect` 文本嵌入
  - `get_current_instruction()`: 返回当前 subgoal
  - `check_and_advance(image)`: 用 SigCLIP 判断当前子步骤是否完成，满足阈值则前进

#### `examples/run_openvla_oft_substep_rl_libero.sh`

Slurm + Apptainer 启动脚本，参考 `backup.sh` 风格编写，新增:
- **Section 5 (Substep RL CONFIG)**: `APD_PLANS_PATH`, `USE_SUBSTEP_RL`, `SIGCLIP_MODEL_PATH`, `SUBSTEP_COMPLETION_THRESHOLD`, `CONTRASTIVE_SAMPLE_INTERVAL`, `VERIFIER_REWARD_COEF`, `CONTRASTIVE_REWARD_COEF`, `KL_COEF`
- **Section 8 (Validation)**: 检查 `APD_PLANS_PATH` 合法性
- **Section 10 (INNER_CMD)**: 传递所有 substep RL 相关 Hydra override
- `open_clip_torch` 使用 `--no-deps` 安装，避免 CPU torch 污染

### 3.2 修改文件

#### `verl/workers/rollout/rob_rollout.py`

**`RobHFRollout.__init__`** (约 line 467-480):
- 根据 `config.use_substep_rl` 条件加载 `APDPlanManager` 和 `SigCLIPRewardModel`
- 初始化 `substep_threshold` 和 `contrastive_interval`

```python
self.use_substep_rl = getattr(config, 'use_substep_rl', False)
self.apd_manager = None
self.sigclip_model = None
if self.use_substep_rl:
    from verl.utils.substep_reward import APDPlanManager, SigCLIPRewardModel
    apd_path = config.apd_plans_path
    sigclip_path = getattr(config, 'sigclip_model_path', 'timm/ViT-B-16-SigLIP-256')
    self.substep_threshold = getattr(config, 'substep_completion_threshold', 0.25)
    self.contrastive_interval = getattr(config, 'contrastive_sample_interval', 16)
    self.apd_manager = APDPlanManager(apd_path)
    self.sigclip_model = SigCLIPRewardModel(sigclip_path, device=torch.device('cuda'))
```

**`_generate_minibatch_libero`** (约 line 830-1001):
- `task_records` 初始化新增 `contrastive_score` 和 `contrastive_count` 字段
- 训练 rollout 时启用 `SubstepTracker`，将 VLA 输入指令替换为当前 substep subgoal
- 每 K 步执行对比 forward pass，累积 contrastive score
- 环境返回新观测后，调用 `substep_tracker.check_and_advance()` 判断是否推进子步骤

```python
# 每 K 步: 对比采样
if (use_substep_this_batch and step > 0
        and step % self.contrastive_interval == 0):
    # 1. 采样错误 substep 指令
    wrong_sub = self.apd_manager.sample_wrong_substep(suite, current_sub)
    # 2. 用错误指令做 forward pass
    wrong_output = self._generate_one_step(wrong_input)
    # 3. 计算动作差异
    diff = torch.norm(actions[idx] - wrong_actions[idx]) / sqrt(dim)
    task_records[idx]['contrastive_score'] += min(diff, 1.0)
    task_records[idx]['contrastive_count'] += 1
```

**`_prepare_output_batch`** (约 line 1027-1035):
- 对每个 episode 计算 `contrastive_score / contrastive_count` 平均值
- 作为标量 tensor 写入 batch 的 `"contrastive_score"` 字段

#### `verl/trainer/main_ppo.py`

**`RobRewardManager.__call__`** (约 line 87-97):
- 新增 R_contrastive 奖励处理逻辑
- 从 batch 中取 `contrastive_score`，乘以权重系数后放在最后一个有效 token 位置
- 新增 WandB metrics: `contrastive`（加权后均值）, `contrastive_raw_mean`（原始均值）

```python
contrastive_coef = getattr(self.config.verifier, 'contrastive_reward_coef', 0)
if 'contrastive_score' in data.batch and contrastive_coef != 0:
    contrastive_reward = torch.zeros_like(reward_tensor)
    contrastive_scores = data.batch['contrastive_score'].cpu().numpy().tolist()
    for i in range(contrastive_reward.shape[0]):
        contrastive_reward[i, valid_response_length[i]-1] += contrastive_scores[i]
    reward_tensor += contrastive_coef * contrastive_reward
```

#### `verl/trainer/config/ppo_trainer.yaml`

新增配置项（均有安全默认值，不影响原有行为）:

```yaml
# rollout 部分
actor_rollout_ref.rollout:
  use_substep_rl: false               # 总开关
  apd_plans_path: null                 # APD_plans.json 路径
  sigclip_model_path: "timm/ViT-B-16-SigLIP-256"
  substep_completion_threshold: 0.25   # SigCLIP 子步骤完成阈值
  contrastive_sample_interval: 16      # 每 K 步做一次对比采样

# verifier 部分
verifier:
  contrastive_reward_coef: 0           # 默认 0，不影响原有行为
```

---

## 4. 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 子步骤切换策略 | APD 预定义 + SigCLIP | 复用 openvla-oft 已验证的 APD 计划和 SigLIP 模型，无需额外训练 |
| 错误指令来源 | 同 suite 其他任务的 substep | 保证语义合理性（同类操作场景），避免跨 suite 的无意义指令 |
| 优势估计 | GRPO (不变) | 保持与原始 SimpleVLA-RL 一致，GRPO 对稀疏奖励有归一化优势 |
| R_substep (子步骤进度奖励) | 不使用 | 避免过度依赖 SigCLIP 的子步骤判断噪声，R_contrastive 已提供足够的指令敏感信号 |
| 对比采样频率 K | 16 (约每 2 个 VLA step) | 平衡计算开销与信号密度。整个 batch 的对比 forward pass 可复用已有的 VLA 推理管线 |
| R_task 权重 w1 | 5 | 与原始 SimpleVLA-RL 一致 |
| R_contrastive 权重 w2 | 2 | 经验值，确保对比信号有影响力但不压过任务奖励 |
| KL 系数 | 0.00 | 与原始 SimpleVLA-RL 一致，不使用 KL 惩罚 |

---

## 5. 向后兼容性

所有新功能通过配置开关控制：

- `use_substep_rl: false` → 不加载 APD/SigCLIP，不执行对比采样，rollout 行为与原版完全一致
- `contrastive_reward_coef: 0` → reward manager 跳过 R_contrastive 计算，reward 函数与原版一致

默认配置 (`ppo_trainer.yaml`) 中所有新参数的默认值都设为不启用状态，**不修改任何原有运行配置即可正常使用原始 SimpleVLA-RL**。

---

## 6. 使用方法

### 6.1 前置准备

1. 训练好 substep SFT checkpoint (使用 `finetune_substep.py`)
2. 准备 `APD_plans.json`（从 openvla-oft 项目获取）

### 6.2 启动 IG-RL 训练

```bash
cd SimpleVLA-RL

SFT_MODEL_PATH=/path/to/substep-sft-checkpoint \
APD_PLANS_PATH=/path/to/APD_plans.json \
CKPT_PATH=/path/to/save/rl-checkpoints \
sbatch examples/run_openvla_oft_substep_rl_libero.sh
```

### 6.3 关键超参数

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| 启用 substep RL | `USE_SUBSTEP_RL` | `True` | 总开关 |
| APD 计划路径 | `APD_PLANS_PATH` | (必填) | APD_plans.json 路径 |
| SigCLIP 模型 | `SIGCLIP_MODEL_PATH` | `timm/ViT-B-16-SigLIP-256` | 子步骤完成检测模型 |
| 完成阈值 | `SUBSTEP_COMPLETION_THRESHOLD` | `0.25` | SigCLIP 相似度阈值 |
| 对比采样间隔 | `CONTRASTIVE_SAMPLE_INTERVAL` | `16` | 每 K 环境步采样一次 |
| 任务奖励权重 | `VERIFIER_REWARD_COEF` | `5` | R_task 系数 |
| 对比奖励权重 | `CONTRASTIVE_REWARD_COEF` | `2` | R_contrastive 系数 |
| KL 系数 | `KL_COEF` | `0.00` | KL 散度惩罚 |

### 6.4 WandB 监控指标

| 指标 | 含义 |
|------|------|
| `reward/verifier` | R_task 均值 (即任务成功率 × w1) |
| `reward/contrastive` | R_contrastive 加权均值 |
| `reward/contrastive_raw_mean` | R_contrastive 原始均值 (未加权) |
| `reward/reward_all` | 总 reward 均值 |

---

## 7. 文件变更清单

```
SimpleVLA-RL/
├── verl/
│   ├── utils/
│   │   └── substep_reward.py              [新增] APDPlanManager, SigCLIPRewardModel, SubstepTracker
│   ├── trainer/
│   │   ├── main_ppo.py                    [修改] RobRewardManager 新增 R_contrastive
│   │   └── config/
│   │       └── ppo_trainer.yaml           [修改] 新增 substep RL 配置项
│   └── workers/
│       └── rollout/
│           └── rob_rollout.py             [修改] RobHFRollout 集成子步骤切换和对比采样
├── examples/
│   └── run_openvla_oft_substep_rl_libero.sh [新增] Slurm+Apptainer 启动脚本
└── docs/
    └── substep_rl_design.md               [新增] 本文档
```
