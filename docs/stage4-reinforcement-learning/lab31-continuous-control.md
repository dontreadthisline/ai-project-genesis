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

<iframe src="resources/figures/algo-evolution.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="DDPG → TD3 → SAC 的阶梯"></iframe>

<p align="center"><b>图 31.2</b>：三算法的多 seed 稳定性</p>

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

<iframe src="resources/figures/sac-entropy.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="最大熵：付钱买探索"></iframe>

<p align="center"><b>图 31.1</b>：熵-回报的权衡（α 扫描）</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `DeterministicActor` — -> tanh -> [-2, 2]
- `DoubleCritic` — Q1, Q2
- `ddpg_td3_step` — cfg 切换三处修复
- `SACActor` — 高斯 + tanh 压缩（log_prob 的 tanh 修正项）
- `sac_step` — sac_step

- 接口骨架：[`../../code/lab31-continuous-control/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab31-continuous-control/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab31-continuous-control/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab31-continuous-control/selftest.py)
- 实验入口：`cd code/lab31-continuous-control && uv sync && uv run run_lab31.py`


