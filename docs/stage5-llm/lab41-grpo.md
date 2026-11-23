---
title: "Lab 41：简化 GRPO"
lab: 41
stage: 5
---

对应大纲实验 41。任务：可自动验证的算术问答（如两位数加减乘）。Lab 30 的 PPO 机制 + Lab 03 的组统计在此汇合，Lab 45 的 Agent RL 是它的直接扩展。

## 0. 理论问题

1. GRPO 用组内相对基线替代 value network，为什么可行？
2. 组大小 G 与奖励噪声如何权衡？
3. KL 正则的「相对谁」在 GRPO 中如何变化？

## 1. 数学基础

### 1.1 从 PPO 到 GRPO

PPO 的优势来自 GAE（需要 critic，Lab 30 §1.3）。GRPO 的替代：同一 prompt $x$ 采样 $G$ 个回答 $\{y_1..y_G\}$，用组内统计归一化奖励作优势：

$$
A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G}) + \epsilon}
$$

直觉：组均值是该 prompt 难度的自适应基线（Lab 28 baseline 思想的按组实例化）。可验证奖励 $r_i \in \{0, 1\}$（对/错）：组内全对或全错时 $A \equiv 0$（无信号，样本弃用——梯度为零，天然课程效应：太简单/太难的任务不产生更新）。

目标（token 级 PPO 式）：

$$
J = \mathbb E\left[ \frac{1}{|y_i|}\sum_t \min\big( \rho_{i,t} A_i,\ \mathrm{clip}(\rho_{i,t}, 1\pm\epsilon) A_i \big) \right] - \beta\, \mathbb E\left[ D_{KL}(\pi_\theta \| \pi_{ref}) \right]
$$

$$
\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{old}(y_{i,t} \mid x, y_{i,<t})}
$$

重要性比值与 clip 语义逐 token（序列优势广播到每个 token，按长度归一）。

### 1.2 与 PPO 的逐项对照

| | PPO（Lab 30） | GRPO |
|---|---|---|
| 优势来源 | GAE（critic） | 组内归一化（无 critic） |
| 动作 | 环境动作 | token |
| KL 参考 | $\pi_{old}$（可选） | $\pi_{ref}$（SFT 模型，固定） |
| 奖励 | 环境标量 | 可验证判分（0/1） |

KL 参考从 $\pi_{old}$ 换成 $\pi_{ref}$：约束对象是「别离 SFT 太远」而非「别离上一轮太远」——语言任务的漂移累积更需要绝对锚点（Lab 39 的 DPO 用同一个 ref）。

### 1.3 组大小与信号

奖励 0/1 时组内至少一个对一个错才有梯度。难度 $p$（策略正确率）下，组内有信号的 probability $= 1 - p^G - (1-p)^G$。$G$ 小于此概率骤降；$G$ 大则采样成本线性增。最优训练区间 $p \in (0.2, 0.8)$：学习发生带——GRPO 的隐式课程。

## 2. 实现任务

<div class="theorem algorithm">
<div class="theorem-title">算法 41.1（GRPO）</div>

1. **for** prompt $x \sim \mathcal D$ **：**
   1. 采样组 $\{y_1, \dots, y_G\} \sim \pi_\theta(\cdot \mid x)$，逐 token 记录 $\log\pi_{old}$。
   2. 可验证判分 $r_i = R(x, y_i) \in \{0, 1\}$。
   3. 组内归一化优势 $\hat A_i = \dfrac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \epsilon}$；全对/全错组 $\hat A \equiv 0$，弃用。
   4. **for** epoch **：** 逐 token 比值 $\rho$，梯度步
      $$\nabla_\theta \left[ \frac{1}{|y|} \sum_t \min\!\big(\rho_t \hat A,\ \mathrm{clip}(\rho_t, 1 \pm \epsilon)\, \hat A\big) - \beta\, \mathrm{KL}(\pi_\theta \| \pi_{ref}) \right]$$

</div>

```python
def make_arith_task(): ...             # 题目 + 可验证判分器
def group_rollout(model, prompts, G, cfg): ...   # 批采样 G 个回答 + logprob_old
def grpo_advantage(rewards): ...          # 1.1 归一化
def grpo_loss(...): ...                   # token 级 clip 目标 + KL(ref)
```

自测：

```python
# T1: 归一化性质：A 的组内均值 == 0；全同奖励时 A 全 0
# T2: 判分器：构造对/错回答，r ∈ {0,1} 边界情况（大小写/多空格）明确
# T3: token 级 ratio：更新前全 1；一步更新后直方图分散
# T4: KL(ref) 单调性：β 大时训练后策略与 ref 的 KL 显著更小
# T5: 无信号组弃用：全对/全错组对参数更新贡献为 0（梯度检查）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | GRPO vs 简化 PPO（带 critic） | 同预算 | 准确率曲线、参数/内存 |
| E2 | $G \in \{4, 8, 16\}$ | — | 训练效率（每样本更新的信号量） |
| E3 | $\beta \in \{0, 0.01, 0.1\}$ | — | 准确率 vs 退化（重复/乱码）形态 |
| E4 | 任务难度带扫描（1 位加法 vs 2 位乘法） | — | 组内信号率与学习速度 |
| E5 | 训练中组内奖励分布演化 | — | 全错带 → 学习带 → 全对带 |

## 4. 预期图表

- 图 1：E1 双曲线：GRPO 无 critic 但准确率相当、内存更小。
- 图 2：E2 的「每 prompt 采样数 vs 学习速度」曲线（G=16 慢起步后反超）。
- 图 3：E3 的 β=0 群体：奖励升但生成质量退化（乱码骗过判分器或复读）——KL 正则价值的直接证据。
- 图 4：E5 的堆叠条形图：训练早期大部分组无信号（全错），随训练「学习带」占比先升后降（任务被攻克）。

## 5. 观测指标

`acc_curve`、`kl_to_ref`、`signal_ratio`（有梯度组占比）、`clip_fraction`、`group_reward_dist`。

## 6. 结果分析

1. E5 的信号率曲线如何刻画「隐式课程」？与教学数据排序（curriculum learning）的异同？
2. E3 中 β=0 的退化形态与 reward hacking（Lab 44）有什么联系？
3. 组归一化 vs GAE：各自引入的偏差来源？（组均值噪声 vs critic 逼近误差。）

## 7. 验收清单

- [ ] T1-T5 通过；token 级 ratio、组优势、KL 项全部正确实现
- [ ] 复现「学习带」演化与 β 的制衡
- [ ] 算术任务准确率显著超越 SFT 基线
- [ ] 阶段验收：小型 LLM 的预训练→推理→SFT→LoRA→DPO→RM→GRPO 全链路走通
