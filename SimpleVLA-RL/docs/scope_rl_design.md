# Scope RL 设计文档

## 概览

本文档描述在 `SimpleVLA-RL` GRPO 训练框架中新增的两项功能：

1. **Object Position Swap**：在每个 episode 开始时随机交换物体的桌面位置，迫使模型依赖语言指令而非记忆固定位置。
2. **Gripper-to-Target Distance Reward**：基于夹爪与目标物体的最近距离给予稠密奖励，缓解稀疏奖励导致的学习信号不足。

两项功能均以**零侵入方式**实现：通过 config flag 控制，默认关闭，不影响原有代码路径。

---

## 一、Object Position Swap

### 1.1 设计目标

标准 LIBERO 训练中，物体位置固定，VLA 模型容易退化为"视觉位置记忆"而非理解语言指令。Swap 通过在 episode 开始时随机交换两个物体的 (x, y) 位置，打破位置固定性，强制语言理解。

### 1.2 Swap 策略

**物体选取（两类池）：**

- **targets**：BDDL 文件 `(:obj_of_interest ...)` 列出的任务相关物体（待夹取物、目标容器）
- **others**：场景中其余可移动物体（distractors）

**配对逻辑（优先策略）：**

$$\text{eligible pairs} = \{(t, o) \mid t \in \text{targets},\ o \in \text{others},\ \|p_t - p_o\|_2 \leq d_{\max}\}$$

优先从 eligible pairs 中随机选一对交换。若 eligible 为空（如所有物体均为 target，或无满足距离条件的对），则回落为任意两个可移动物体组成的对中距离满足条件的对。

**什么被交换：**

仅交换 (x, y)，保留 z 和四元数朝向，避免物体穿地板或姿态异常：

$$x_a, y_a \leftrightarrow x_b, y_b \quad (\text{z, qw, qx, qy, qz 不变})$$

MuJoCo 中 free joint（`jnt_type == 0`）的 qpos 布局为 `[x, y, z, qw, qx, qy, qz]`（7值）。

### 1.3 GRPO n_samples 一致性

GRPO 对同一个 prompt 采集 $n$ 个独立 rollout（本项目 `DATA_N_SAMPLES=4`），advantage 为组内归一化：

$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$

若每个 rollout 看到不同的物体摆放，则 $n$ 个 rollout 实际上是不同任务的 rollout，advantage 估计偏差。

**解决方案：** 在父进程为每个唯一 $(task\_name, task\_id, trial\_id)$ 组预计算一个 `swap_seed`，该组所有 $n$ 个子进程传入相同 seed：

```python
key = (task_suite_name[idx], task_id[idx], trial_id[idx])
if key not in swap_seeds:
    swap_seeds[key] = random.randint(0, 2**31 - 1)
```

每个子进程使用 `random.Random(seed)`（独立 RNG，不影响全局随机状态），保证同组所有 rollout 的物体布局完全一致。

### 1.4 距离课程（Swap Distance Curriculum）

目标：训练初期只交换较近的物体（交换幅度小，难度低），随训练进行逐步允许更远的物体对参与交换（更大位移，难度高）。

$$d_{\max}(t) = d_{\text{start}} + \min\!\left(\frac{t}{T_{\text{curriculum}}}, 1\right) \cdot (d_{\text{end}} - d_{\text{start}})$$

其中：
- $d_{\text{start}}$：训练第 0 步允许的最大交换距离（LIBERO 中最近物体对约 0.11–0.12 m，建议 0.14 m）
- $d_{\text{end}}$：课程结束后允许的最大交换距离（建议 0.40 m）
- $T_{\text{curriculum}}$：课程步数（建议 12500 training steps）
- $t$：当前 global training steps

### 1.5 参数配置

| Shell 变量 | YAML key | 默认值 | 含义 |
|---|---|---|---|
| `SWAP_OBJECTS` | `rollout.swap_objects` | `False` | 是否开启 swap |
| `SWAP_DISTANCE_START` | `rollout.swap_min_distance` | `0.12` | 课程起始最大距离（m） |
| `SWAP_DISTANCE_END` | `rollout.swap_max_distance` | `0.40` | 课程终止最大距离（m） |
| `SWAP_CURRICULUM_STEPS` | `rollout.swap_curriculum_steps` | `0` | 课程总步数（0=不开课程，直接用 max） |

---

## 二、Gripper-to-Target Distance Reward

### 2.1 设计动机

GRPO 原有奖励为**纯稀疏**：

$$r_{\text{success}} = R_{\text{coef}} \cdot \mathbf{1}[\text{task complete}] = 5.0 \cdot \{0, 1\}$$

对于成功率接近 0 的难任务（swap 后尤其如此），几乎所有 rollout 奖励为 0，组内方差 $\sigma_r \to 0$，advantage 退化为 0，策略梯度消失，学习停滞。

距离奖励提供稠密引导信号，鼓励夹爪向目标靠近，缓解冷启动问题。

### 2.2 距离度量

**目标物体定义：** 与 Swap 相同，使用 BDDL `(:obj_of_interest ...)` 中的物体对应的 MuJoCo body。多目标任务中同时追踪所有 target bodies。

**Trajectory-level 最近距离：** 取整条轨迹中夹爪到任意目标的最小欧氏距离：

$$d_{\min} = \min_{t=0}^{T} \min_{k \in \text{targets}} \|\text{eef}_t - \text{body}_k\|_2$$

使用 min-over-trajectory 而非 mean，原因：
- mean 距离会惩罚"夹取后运输"阶段（夹爪携带物体远离其他目标），产生错误信号
- min 只奖励"曾经靠近过"，对正常操作路径无负面影响

**成功 padding：** 若 `task complete = True`，强制令 $d_{\min} = 0$，理由：
- 任务成功时机器人必定已抓取并放置目标，$d_{\min}$ 的真实值已不重要
- 避免因不同成功轨迹路径差异导致奖励不稳定
- 成功 rollout 获得距离奖励上界 1.0，加强正样本信号

### 2.3 奖励函数

使用**高斯核**将距离映射到 $[0, 1]$：

$$r_{\text{dist}} = \exp\!\left(-\frac{d_{\min}^2}{2\sigma^2}\right)$$

性质：
- $d_{\min} = 0$（接触/成功）：$r_{\text{dist}} = 1.0$
- $d_{\min} = \sigma$（1 倍宽度）：$r_{\text{dist}} \approx 0.607$
- $d_{\min} = 2\sigma$（2 倍宽度）：$r_{\text{dist}} \approx 0.135$
- $d_{\min} \gg \sigma$：$r_{\text{dist}} \to 0$

$\sigma = 0.05$ m（5 cm），对应 LIBERO 抓取精度量级。

### 2.4 总奖励

距离奖励叠加在稀疏成功奖励之上，注入 **episode 最后一个 action token**：

$$r_{\text{total}} = \lambda_{\text{success}} \cdot r_{\text{success}} + \lambda_{\text{dist}} \cdot r_{\text{dist}}$$

设计约束：

$$\lambda_{\text{dist}} \cdot 1.0 \ll \lambda_{\text{success}} \cdot 1.0$$

即距离奖励的最大值必须远小于成功奖励，保证**成功始终主导学习方向**。

当前取值：$\lambda_{\text{dist}} = 0.3$，$\lambda_{\text{success}} = 5.0$，比值 $= 0.06$，满足约束。

**奖励写入位置：** 与稀疏奖励一致，写在 `reward_tensor[i, valid_response_length[i] - 1]`，即当前轨迹有效 token 的最后一位：

```python
r_dist = torch.exp(-min_dist ** 2 / (2.0 * sigma ** 2))
dist_reward[i, valid_response_length[i] - 1] += r_dist[i]
reward_tensor += dist_reward_coef * dist_reward
```

### 2.5 参数配置

| Shell 变量 | YAML key | 默认值 | 含义 |
|---|---|---|---|
| `DIST_REWARD_COEF` | `verifier.dist_reward_coef` | `0.0` | 距离奖励权重（0=禁用） |
| `DIST_REWARD_SIGMA` | `verifier.dist_reward_sigma` | `0.05` | 高斯核宽度（m） |

---

## 三、数据流总结

```
episode 开始
  └─ (do_swap=True) → _random_swap_objects(seed=swap_seed, max_distance=d_max(t))
       └─ 解析 BDDL obj_of_interest → 构建 (target, distractor) eligible pairs
       └─ 按课程 d_max 过滤 → rng.choice → 交换 (x,y) → sim.forward()

rollout 过程
  └─ 每步 env.step() 后
       └─ _min_dist_to_targets(obs, sim, target_body_ids) → 更新 min_dist
       └─ 若 complete=True → min_dist = 0.0 (success padding)

rollout 结束
  └─ _prepare_output_batch
       └─ batch["min_dist"] = min(min_dist, 10.0)  ← cap at 10m, avoid NaN

RobRewardManager.__call__
  └─ r_success = verifier_reward_coef * complete
  └─ r_dist = dist_reward_coef * exp(-min_dist² / 2σ²)  [if dist_reward_coef > 0]
  └─ reward_tensor[i, last_token] += r_success[i] + r_dist[i]

GRPO advantage
  └─ Â_i = (r_i - mean_group) / (std_group + ε)  [per (task, trial) group]
```

---

## 四、文件修改清单

| 文件 | 修改内容 |
|---|---|
| `libero_env_worker.py` | 新增 `_parse_obj_of_interest`, `_random_swap_objects`, `_get_target_body_ids`, `_min_dist_to_targets`；`env_worker` 增加 `do_swap/swap_seed/swap_max_distance` 参数；添加 min_dist 追踪逻辑 |
| `verl/workers/rollout/rob_rollout.py` | 新增 `_compute_swap_max_distance`；`_generate_minibatch_libero` 中增加 swap seed 预计算、子进程参数传递；`_prepare_output_batch` 输出 `batch["min_dist"]` |
| `verl/trainer/main_ppo.py` | `RobRewardManager.__call__` 中增加距离奖励计算块（`dist_reward_coef > 0` 门控） |
| `verl/trainer/config/ppo_trainer.yaml` | `rollout` 下新增 `swap_objects/swap_min_distance/swap_max_distance/swap_curriculum_steps`；`verifier` 下新增 `dist_reward_coef/dist_reward_sigma` |
| `examples/run_openvla_oft_scope.sh` | 新增 swap 和 dist reward 的 shell 变量默认值及 Hydra 参数透传 |
| `scope_rl_trail.sh` | 新增所有 swap 和 dist reward 的实验参数赋值 |
