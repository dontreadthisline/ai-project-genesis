---
title: "Lab 29：Actor-Critic"
lab: 29
stage: 4
---

对应大纲实验 29。环境：CartPole。依赖 Lab 28。

## 0. 理论问题

1. 用学到的 $V^\pi$ 做 baseline，比批内统计好在哪里？
2. TD 误差形式的 advantage 兼顾了什么？
3. Actor 和 Critic 的学习率、更新频率为什么需要平衡？

## 1. 数学基础

### 1.1 优势函数

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s), \qquad \mathbb E_{a\sim\pi}[A^\pi(s,a)] = 0
$$

策略梯度的优势形式：

$$
\nabla_\theta J = \mathbb E\left[ \sum_t A^\pi(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$

（由 $Q = A + V$ 与 baseline 无偏性直接得到。）直觉：$A > 0$ 的动作比该状态下平均更好，提高其概率；$A < 0$ 降低。逐状态去中心比全局 baseline 更精细。

### 1.2 TD 误差作为优势的估计

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t), \qquad \mathbb E_{\pi}[\delta_t \mid s_t, a_t] = Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)
$$

（第二式用 $V$ 的贝尔曼方程展开即得。）$\delta_t$ 只需一次 $V$ 前向（而非两条轨迹），是优势的单步无偏（给定当前 $V$）估计——方差比 $G_t - V$ 低，代价是引入 $V$ 估计的偏差。Critic 损失：

$$
\mathcal L_V = \mathbb E\left[ (r + \gamma V(s'; \phi) - V(s; \phi))^2 \right] \quad (\text{同 Lab 26 的 target 病，可加目标网络或小 } \tau)
$$

### 1.3 联合训练

Actor 用 $-\log\pi \cdot \delta.\text{detach}()$，critic 用 $\delta^2$。两者耦合的非平稳性：critic 落后则 actor 用错误优势，actor 快变则 critic 追不上分布。实践：critic lr 常大于 actor lr、或 critic 每步多次更新。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `ActorCritic` — 共享 trunk：策略头 logits + 价值头标量
- `a2c_train` — on-policy：采集 n_step 回合批，一次更新

- 接口骨架：[`../../code/lab29-actor-critic/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab29-actor-critic/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab29-actor-critic/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab29-actor-critic/selftest.py)
- 实验入口：`cd code/lab29-actor-critic && uv sync && uv run run_lab29.py`


