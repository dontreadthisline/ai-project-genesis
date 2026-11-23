---
title: "Lab 30：实现 PPO"
lab: 30
stage: 4
---

对应大纲实验 30。环境：CartPole。依赖 Lab 29。本实验的机制将原样迁移到 Lab 41（GRPO）与 Lab 45（Agent RL）。

## 0. 理论问题

1. 重要性采样如何让旧数据被复用？比值出现在哪一步？
2. Clipped 目标在限制什么？它与 KL 惩罚的差别？
3. GAE 的 $\lambda$ 如何在偏差与方差间插值？

## 1. 数学基础

### 1.1 重要性采样比

期望的分布替换恒等式（Lab 03 选做实验的正用）：

$$
\mathbb E_{a \sim \pi_{old}}[f(a)] = \mathbb E_{a \sim \pi_{new}}\left[ \frac{\pi_{old}(a)}{\pi_{new}(a)} f(a) \right]
\ \Longleftrightarrow\
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

数据由 $\pi_{old}$ 采集，用它多次更新 $\pi_\theta$：策略梯度的重要性修正版 $\nabla J = \mathbb E[r_t(\theta) A_t \nabla\log\pi_\theta]$。$r$ 偏离 1 的程度 = 新旧策略偏离度，正是要限制的对象。

### 1.2 Clipped 目标

$$
J^{CLIP}(\theta) = \mathbb E_t\left[ \min\big( r_t A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \big) \right]
$$

逐样本取 min 的含义：$A > 0$（好动作）时上限 $1+\epsilon$——好动作概率提得过高就不再奖励；$A < 0$ 时下限 $1-\epsilon$——坏动作压得过低就不再受益。**只惩罚「越界后还继续同向移动」**，下界保证目标函数是真实代理目标的悲观下界（TRPO 信任域的一阶近似）。

### 1.3 GAE（推导骨架）

n-step 优势 $\hat A^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l+1} + \gamma^n V(s_{t+n}) - V(s_t)$，即 TD 误差的加权和：$\hat A^{(n)} = \sum_{l=0}^{n-1} (\gamma\lambda)^l$ 型组合。GAE 定义为各 n-step 的指数加权：

$$
\hat A^{GAE(\gamma,\lambda)}_t = \sum_{l\ge 0} (\gamma\lambda)^l \delta_{t+l+1}, \qquad \delta_t = r_t + \gamma V(s_t) - V(s_{t-1})
$$

$\lambda = 0$ 退化为单步 $\delta$（低方差高偏差）；$\lambda = 1$ 等价 $G_t - V$（MC，无 $V$ 偏差高方差）。$\lambda$ 是 Lab 25 TD/MC 光谱的连续插值参数。

### 1.4 完整目标

$$
\mathcal L = \mathbb E\left[ J^{CLIP} - c_v (V - \hat R)^2 + c_e \mathcal H(\pi_\theta) \right] - \beta\, \mathbb E[\mathrm{KL}(\pi_\theta \| \pi_{old})]
$$

值损失、熵奖励（保持探索，防过早坍缩）、KL 项（软约束，可做早停：$\mathrm{KL} > 1.5 \times \mathrm{KL}_{target}$ 跳过该批更新）。

## 2. 实现任务

<div class="theorem algorithm">
<div class="theorem-title">算法 30.1（PPO）</div>

1. **for** 迭代 $= 1, 2, \dots$ **：**
   1. 用 $\pi_{old}$ 采集 batch $\{(s_t, a_t, r_t)\}$，记录 $\log\pi_{old}(a_t \mid s_t)$。
   2. 计算 GAE 优势 $\hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l+1}$ 与回报目标 $\hat R_t$。
   3. **for** epoch $= 1, \dots, K$ **：**
      - 比值 $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{old}(a_t \mid s_t)$；
      - 若 $\mathrm{KL}(\pi_\theta \| \pi_{old}) > 1.5\, \mathrm{KL}_{target}$：**break**（信任域早停）；
      - 梯度步：$\nabla_\theta\left[ J^{CLIP} - c_v (V_\phi - \hat R)^2 + c_e \mathcal H(\pi_\theta) \right]$。

</div>

```python
class ActorCritic: ...                    # Lab 29 复用
def gae(rewards, values, dones, gamma, lam): ...
def ppo_loss(logits_new, logits_old, actions, adv, returns, cfg): ...
def ppo_train(env_fn, iters, cfg): ...    # 采集 batch -> PPO epoch 多轮更新
```

自测：

```python
# T1: 更新前 r == 1 且 clip 不变：初始时刻目标 == A 的均值
# T2: clip 语义数值：r=2, A>0 时目标锁定 (1+ε)A；r=0.2, A<0 时锁定 (1-ε)A
# T3: GAE 端点：λ=0 时 == δ；λ=1 且 V=0 时 == G_t（手算小例）
# T4: advantage 标准化后均值 0、方差 1（批内）
# T5: KL 早停：构造大策略变化，KL 超阈值时更新被跳过
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | PPO vs A2C vs REINFORCE | 同种子组 | 稳定性、样本效率 |
| E2 | $\epsilon \in \{0.05, 0.2, 0.5\}$ | — | ratio 分布、策略 KL 曲线 |
| E3 | $\lambda \in \{0, 0.5, 0.95, 1\}$ | — | 收敛稳定性 |
| E4 | PPO epoch 数 $\{1, 4, 10, 20\}$ | — | 复用过度：KL 爆炸与性能回退 |
| E5 | 去掉 entropy 项 | — | 过早坍缩的形态 |

## 4. 预期图表

- 图 1：三算法回报曲线：PPO 的曲线平滑且无 REINFORCE 的间歇性下探。
- 图 2：E2 中 ratio 直方图随 epoch 演化：ε=0.05 的分布在 1±0.05 处削顶堆积。
- 图 3：E3 的 λ 光谱稳定性排序。
- 图 4：E4 的 KL 曲线：epoch 过大时 KL 超限、回报回退（信任域破坏的可视化）。

## 5. 观测指标

`return_curves`、`ratio_hist`、`approx_kl_curve`、`clip_fraction`。

## 6. 结果分析

1. clip_fraction（被截断的样本比例）健康区间是多少？过高/过低各说明什么？
2. E4 找到本设置下的最大可用 epoch——它与 ε 的关系符合信任域直觉吗？
3. 为什么 PPO 在 LLM 后训练（Lab 41/45）成为默认选择而非 Q-Learning？（策略约束 + token 级重要性比值的可行性。）

## 7. 验收清单

- [ ] T1-T5 通过；能白板推导重要性比、clipped 目标的分段语义、GAE 加权和
- [ ] 完成 λ 与 epoch 的两项消融并解释曲线
- [ ] CartPole eval 回报 $\geq 475$ 且曲线无塌陷
