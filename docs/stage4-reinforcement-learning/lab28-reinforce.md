---
title: "Lab 28：REINFORCE"
lab: 28
stage: 4
---

对应大纲实验 28。环境：CartPole。Lab 03（score function 估计）+ Lab 16（两种梯度路线）在此汇合。

## 0. 理论问题

1. 策略梯度定理如何把「回报的梯度」变成「期望的梯度」？
2. 为什么减一个 baseline 不改变期望却降低方差？
3. 策略梯度与 Q-Learning 这条值方法路线的根本差别？

## 1. 数学基础

### 1.1 策略梯度定理（推导）

目标 $J(\theta) = \mathbb E_{\tau \sim \pi_\theta}[R(\tau)]$。轨迹似然 $\pi_\theta(\tau) = p(s_0)\prod_t \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)$。log-derivative 恒等式（Lab 16 §1.1）：

$$
\nabla_\theta J = \mathbb E_{\tau}\left[ R(\tau)\, \nabla_\theta \log \pi_\theta(\tau) \right]
= \mathbb E_{\tau}\left[ R(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

环境动力学 $P$ 与 $\theta$ 无关，$\nabla \log P = 0$ 项消失——**不需要环境模型**。因果性修剪（未来动作不影响过去奖励）得每时步形式：

$$
\nabla_\theta J = \mathbb E\left[ \sum_t G_t\, \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$

单轨迹 MC 估计即可更新（Lab 03：无偏、方差随回合长度增长）。

### 1.2 Baseline 的无偏性（推导）

对任意与动作无关的 $b(s_t)$：

$$
\mathbb E_{a \sim \pi}\left[ b(s)\, \nabla_\theta \log \pi_\theta(a|s) \right] = b(s)\, \nabla_\theta \sum_a \pi_\theta(a|s) = b(s)\, \nabla_\theta 1 = 0
$$

故减 baseline 不改变期望梯度，只改变估计量分布。取 $b(s) \approx V^\pi(s)$ 时 $G_t - b$ 是信号的「去中心化」：回报恒正的问题（如每步 +1）不再把所有动作的概率一律推高——方差大幅下降。

### 1.3 离散策略网络

$\pi_\theta(a|s) = \mathrm{softmax}(f_\theta(s))$，$\nabla_\theta \log \pi$ 即 softmax 交叉熵对 logits 的梯度减去动作独热（Lab 06 §1.2 同款结构）。

## 2. 实现任务

```python
class PolicyNet: ...                       # 4 -> 128 -> 128 -> 4 logits
def reinforce(env_fn, episodes, lr, gamma, use_baseline, rng): ...
def policy_grad_loss(logits, actions, returns_or_adv): ...   # -logπ * A 的均值
```

自测：

```python
# T1: baseline 无偏性数值验证：固定状态、随机 b，E[b ∇logπ] 随采样数 → 0
#     （1e5 样本，均值范数 < 1e-2 × 无 b 时的梯度范数）
# T2: 梯度方向检查：增大高回报动作的 logπ 概率（单步手工例）
# T3: 方差对照：同一批轨迹，带/不带 baseline 的梯度估计样本方差比值
# T4: 收敛后策略接近确定性：eval 时动作熵 < 0.3 nats
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 无 baseline / 常数 baseline / 批内标准化 | 同种子组 | 回报曲线、梯度方差 |
| E2 | 学习率 $\{0.001, 0.01, 0.1\}$ | — | 崩溃/稳定/过慢 |
| E3 | $\gamma \in \{0.95, 0.99, 1.0\}$ | — | 学到的偏好与方差 |
| E4 | 每 episode 更新 vs 批 10 回合更新 | — | 梯度噪声（连 Lab 03/08） |

## 4. 预期图表

- 图 1：E1 三条回报曲线：无 baseline 收敛最慢且震荡大（CartPole 回报恒正，未中心化）；批标准化最优。
- 图 2：梯度范数的分布直方图随 baseline 变化（宽度收窄一个量级）。
- 图 3：E2 学习率敏感性：REINFORCE 的对步长的高敏感（与 Lab 07 呼应）。
- 图 4：训练中策略熵曲线：从 ln4 ≈ 1.39 降到 0.3 以下。

## 5. 观测指标

`return_curves`、`grad_var_by_baseline`、`final_policy_entropy`。

## 6. 结果分析

1. 用 1.2 的推导说明「为什么 CartPole 这类恒正回报环境特别需要 baseline」。
2. E4 中批更新相当于 Lab 08 的哪个结论？（batch 梯度方差 $\propto 1/b$。）
3. 对比 Lab 26：策略梯度直接优化 $J$、无需 max/自举，代价是什么？（方差高、样本效率低——Lab 29/30 的改进方向。）

## 7. 验收清单

- [ ] T1-T4 通过；能白板完成策略梯度定理与 baseline 无偏性的推导
- [ ] 复现 baseline 的方差缩减效果
- [ ] 能说清策略方法与值方法的取舍及后续演进路线（AC → PPO）
