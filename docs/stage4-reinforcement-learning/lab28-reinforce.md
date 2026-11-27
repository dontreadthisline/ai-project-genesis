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

实现清单（签名与 TODO 见骨架源码）：

- `PolicyNet` — 4 -> 128 -> 128 -> 4 logits
- `reinforce` — -logπ * A 的均值
- `policy_grad_loss` — -logπ * A 的均值

- 接口骨架：[`../../code/lab28-reinforce/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab28-reinforce/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab28-reinforce/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab28-reinforce/selftest.py)
- 实验入口：`cd code/lab28-reinforce && uv sync && uv run run_lab28.py`


