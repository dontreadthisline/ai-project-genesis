---
title: "Lab 45：Agent PPO / GRPO"
lab: 45
stage: 6
---

对应大纲实验 45 + 综合项目。依赖前两个阶段全部：Lab 41 的 GRPO 机制、Lab 42 的环境、Lab 43 的轨迹、Lab 44 的奖励。

## 0. 理论问题

1. 把 GRPO 从「单轮生成」迁移到「多步工具交互」，轨迹分布发生了什么变化？
2. 组内基线在多步任务中还够用吗？
3. 训练后的改善具体发生在哪些行为维度（连 Lab 43 的失败分类树）？

## 1. 数学基础

### 1.1 轨迹分布的目标

$$
J(\theta) = \mathbb E_{\tau \sim \pi_\theta}\!\left[ R(\tau) \right], \qquad
\tau \sim p(x)\prod_{t=1}^{K} \pi_\theta(a_t \mid h_t)\, \mathbb 1[\text{env 确定性}]
$$

与 Lab 41 的差别：一条「样本」从一次生成变为整段多步交互；对数概率是各动作段 token 的联合（观测段的概率不属于策略，条件化掉）：

$$
\log \pi_\theta(\tau \mid x) = \sum_{t=1}^{K} \sum_{j} \log \pi_\theta(a_{t,j} \mid h_t, a_{t,<j})
$$

### 1.2 组相对优势（多步版）

同 prompt 组：$G$ 条完整轨迹 $\tau_1..\tau_G$，$A_i = \frac{R_i - \mathrm{mean}}{\mathrm{std}}$（Lab 41 §1.1 原样），广播到 $\tau_i$ 的全部动作 token。信用分配是轨迹级的：整条对/错，所有动作 token 同权承担——多步信用分配的粗糙近似（精化方向是过程奖励/树搜索，超出主线）。

token 级目标与 KL 项同 Lab 41 §1.1，$\rho_{t,j}$ 逐动作段 token 计。

### 1.3 On-policy 数据的工程约束

rollout 必须用当前（或上一轮）策略采样；轨迹数据用后即弃（off-policy 偏差在多步下被复合放大）。批量 rollout 的吞吐决定训练速度：环境步（工具执行）与模型步（解码）交替，串行慢——并行 worker（独立 env + 批量解码）是必要的工程结构。

### 1.4 训练-评估协议

- 指标：真实成功率（不含 process 奖励）、动作合法率、平均步数、无效动作率、KL to ref、截断率——训练前后按 Lab 43 分类树逐项对照；
- 早停与回滚：成功率不升或 KL 爆炸时回滚 checkpoint（Lab 44 的 hack 监控持续在线）。

## 2. 实现任务

完整闭环七步（大纲 6.3）：

```python
def collect_group_rollouts(model, env, prompts, G, cfg): ...
    # 1.prompt 采样 2.agent rollout 3.轨迹收集（含逐步 logprob）
def compute_group_reward(trajs, reward_cfg): ...     # 4.奖励计算
def compute_advantages(rewards): ...                 # 5.组归一化
def grpo_update(model, ref_model, trajs, advs, cfg): ...  # 6.token 级更新
def evaluate(model, env, tasks, n): ...              # 7.独立评估（真实成功率）
def train_loop(cfg): ...   # 以上循环 + KL 监控 + hack 监控 + checkpoint
```

自测：

```python
# T1: 轨迹 logprob 一致性：rollout 时记录的 logprob_old 与离线重算一致（1e-4）
# T2: 观测段梯度隔离：损失对观测 token 位置的 logprob 无梯度（mask 正确）
# T3: 更新前 ratio 全 1（重放同轨迹计算）
# T4: 组退化：同 prompt 全部轨迹同奖励时，该组参数更新量为 0
# T5: 评估独立性：evaluate 不接触任何奖励权重（真实判分专用）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 训练前 vs 后 | 同任务集 | 成功率/合法率/步数/截断率全面板 |
| E2 | GRPO vs 简化轨迹 PPO（带 critic） | 同预算 | 样本效率、内存 |
| E3 | $G \in \{4, 8, 16\}$ | — | 学习速度 vs 采样成本 |
| E4 | $\beta$（KL）扫描 | — | 成功率 vs 退化形态（连 Lab 41 E3） |
| E5 | 失败分类树：训练前后各类占比 | — | RL 修复了哪一支 |
| E6 | 持出任务集（未见过的题目变体） | — | 泛化 vs 记忆 |

## 4. 预期图表

- 图 1：训练曲线全家福：真实成功率、代理奖励、KL、clip fraction、熵随迭代。
- 图 2：E1 的雷达图（五指标前后对比）。
- 图 3：E5 的分类树占比迁移：截断型与循环型占比大幅下降、协议型几乎不动。
- 图 4：E6 持出集对照：成功率提升的保留比例（泛化系数）。
- 图 5：训练前后同一任务的轨迹对照（定性证据：从空转到有效工具链）。

## 5. 观测指标

`success_before_after`、`failure_taxonomy_shift`、`kl_curve`、`generalization_ratio`、`steps_efficiency`。

## 6. 结果分析

1. E5 中 RL 修复最彻底与最顽固的失败类型各是什么？与「轨迹级信用分配」的粗糙性有什么对应？
2. E6 的泛化系数多少？结合训练任务多样性分析。
3. 组基线 vs critic（E2）：多步场景下两者的方差-偏差表现与 Lab 41 的单步结论一致吗？

## 7. 验收清单（综合项目验收）

- [ ] T1-T5 通过；七步闭环全部自实现，on-policy 数据流正确
- [ ] 真实成功率相对 SFT 基线提升 $\geq 15$ 个百分点（构造任务族）
- [ ] 训练过程可分析：KL、clip、熵、失败迁移四组证据齐全
- [ ] 能完整叙述：概率建模（组统计/轨迹分布）→ 策略梯度（token 级 GRPO）→ 奖励设计（可验证+防 hack）→ 语言模型（SFT 起点）在本项目中各自的贡献——课程主线的终点检验

## 8. 课程终点

至此六阶段闭环：从 Lab 01 的高斯密度到本实验的多步工具交互 RL，全部数学与工程链条贯通。选修方向（Model-based RL、World Model、Planning、Multi-Agent）以当前能力为起点，按需展开。
