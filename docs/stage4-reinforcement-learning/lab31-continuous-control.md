---
title: "Lab 31：Pendulum 连续控制"
lab: 31
stage: 4
---

对应大纲实验 31。环境：Gymnasium Pendulum-v1（连续动作 $a \in [-2, 2]$）。主线：DDPG → TD3 → SAC。

## 0. 理论问题

1. 连续动作下 $\max_a Q$ 不可枚举，确定性策略如何绕过？
2. TD3 针对连续控制的哪三个病？
3. 最大熵 RL 改变了优化目标，带来什么行为差异？

## 1. 数学基础

### 1.1 DDPG：确定性策略梯度

策略 $\mu_\theta(s) \in \mathbb R^{|A|}$（tanh 缩放到动作域）。目标 $J = \mathbb E[Q(s, \mu_\theta(s))]$，链式法则：

$$
\nabla_\theta J = \mathbb E\left[ \nabla_a Q(s, a)\big|_{a=\mu(s)} \cdot \nabla_\theta \mu_\theta(s) \right]
$$

critic 用 Q-Learning 型目标（replay + target 网络，Lab 26 全套）。**确定性策略没有探索**：动作加 OU/高斯噪声训练。on-policy 熵没了，exploration 与利用完全外包给噪声进程。

### 1.2 TD3 的三处修复

1. **双 Q 取 min**：$y = r + \gamma \min(Q_1', Q_2')$——两个独立过估计的 min 抑制 Lab 27 的 max 偏差（连续 max 的内最大化由 actor 目标承担，噪声更大）。
2. **延迟策略更新**：actor 每 d 步更新一次（critic 收敛前别追着跑）。
3. **目标平滑**：目标动作加截断噪声 $\tilde a' = \mu'(s') + \mathrm{clip}(\epsilon, -c, c)$，防 critic 在尖峰处被 actor 利用。

### 1.3 SAC：最大熵 RL

$$
J(\pi) = \mathbb E\left[ \sum_t r_t + \alpha\, \mathcal H(\pi(\cdot|s_t)) \right]
$$

策略改为随机（重参数化采样 $a = f_\theta(s, \epsilon)$，Lab 15/16 的 pathwise 梯度），目标中显式奖励熵。核心收益：**探索内置**（策略保持分布而非噪声外包）、鲁棒性（多模态最优时保持多模态而非坍缩到单点）、温度 $\alpha$ 可自动调节（约束熵下限，$\alpha$ 梯度上升对偶优化；本实验用固定 α 做对照即可）。

三算法演进线：DDPG（能跑但不稳）→ TD3（修值估计）→ SAC（换目标函数，稳定 + 探索内建）。

## 2. 实现任务

```python
class DeterministicActor: ...   # -> tanh -> [-2, 2]
class DoubleCritic: ...         # Q1, Q2
def ddpg_td3_step(batch, nets, cfg): ...    # cfg 切换三处修复
class SACActor: ...             # 高斯 + tanh 压缩（log_prob 的 tanh 修正项）
def sac_step(batch, nets, alpha, cfg): ...
```

自测：

```python
# T1: 确定性策略梯度与数值梯度一致（rel err < 1e-4，小网络）
# T2: 双 Q min 的减偏：对照单 Q 目标，训练后 Q 估计 vs 实际回报偏差
# T3: tanh 压缩 log_prob：包含 -Σ log(1 - tanh(u)^2) 修正项（无此项则概率不归一）
# T4: 延迟更新计数：actor 参数在非更新步严格不变
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | DDPG / TD3 / SAC | 同种子组 | 回报曲线、稳定性（多 seed 方差带） |
| E2 | α $\in \{0.05, 0.2, 0.5\}$（SAC） | — | 策略熵 vs 回报的权衡 |
| E3 | 去掉 TD3 三修复的单项 | — | 各自的病征 |
| E4 | 确定性（DDPG）vs 随机（SAC）的探索轨迹可视化 | — | 相空间覆盖 |

## 4. 预期图表

- 图 1：三算法多 seed 回归带：DDPG 方差大且部分 seed 失败；SAC 最稳。
- 图 2：E2 熵-回报双轴：α 大熵高回报略降（能量花在探索）。
- 图 3：E3 消融：去双 Q → Q 值上飘；去延迟 → actor 追逐噪声 critic；去平滑 → 动作高频抖动。
- 图 4：摆的相空间 (θ, θ̇) 轨迹：SAC 覆盖广（熵驱动），DDPG 收敛后窄轨道。

## 5. 观测指标

`return_bands`（5 seeds）、`policy_entropy`、`q_gap`（Q 均值 - 回报均值）。

## 6. 结果分析

1. E1 中 DDPG 的失败 seed 的 Q 曲线什么样？（值估计发散 → 策略奔向虚高区。）
2. E2 的最优 α 随训练阶段变化吗？（探索需求递减 → 自动 α 的动机。）
3. 为什么 SAC 的随机策略在 LLM 阶段不直接使用，但它的熵思想出现在 GRPO 的熵正则中？（离散 token 的分布策略本身就是随机策略。）

## 7. 验收清单

- [ ] T1-T4 通过；能推导确定性策略梯度与 tanh 修正的 log_prob
- [ ] 完成 DDPG→TD3→SAC 的阶梯实验并解释每步修复
- [ ] Pendulum 回报：SAC $\geq -160$（5 seed 均值）
- [ ] 阶段验收汇总：tabular Q / DQN / REINFORCE / PPO / SAC 全部独立实现过
