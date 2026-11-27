---
title: "Lab 07：优化器对照"
lab: 7
stage: 2
---

对应大纲实验 7。依赖 Lab 06 的 MLP 与数据加载。

## 0. 理论问题

1. Momentum 为什么能加速在病态曲率方向上的收敛？
2. Adam 的偏差修正解决什么问题？AdamW 与 Adam 的差别为什么重要？
3. 学习率调度如何影响最终性能？

## 1. 数学基础

### 1.1 SGD 与 Momentum

SGD：$\theta_{t+1} = \theta_t - \eta g_t$。Momentum 引入速度：

$$
v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

展开递推：$v_t = \sum_{i \le t} \beta^{t-i} g_i$，即历史梯度的指数加权平均。在一致方向的分量按 $\frac{1}{1-\beta}$ 累积放大，震荡方向的分量正负抵消——等效于对损失曲面两个主轴方向使用不同的有效学习率。

### 1.2 Adam 与偏差修正（推导）

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

初期 $m_0 = v_0 = 0$ 导致 $m_t, v_t$ 系统性偏小。以 $m$ 为例，若梯度恒定 $g$，则 $m_t = g(1 - \beta_1^t)$，即 $m_t$ 是 $g$ 的 $(1-\beta_1^t)$ 倍。修正：

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

常数梯度下 $\hat m / \sqrt{\hat v} \approx 1$：有效步长约为 $\eta$，与梯度尺度无关（逐参数自适应）。

### 1.3 AdamW：解耦权重衰减

Adam 将 L2 正则并入梯度（$g + \lambda\theta$）后，$\theta$ 会经过 $1/\sqrt{\hat v}$ 的自适应缩放，正则强度在不同参数上不一致。AdamW 直接解耦：

$$
\theta_{t+1} = \theta_t - \eta\left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t \right)
$$

衰减项不经过自适应缩放。这是 Lab 05 MAP 视角（L2 = 高斯先验）在自适应优化器上的正确延续。

### 1.4 学习率调度

Warmup 线性升温 + 余弦退火：

$$
\eta(t) = \eta_{max} \cdot \min\!\left(\frac{t}{T_w},\ \frac12\left(1 + \cos\frac{\pi (t - T_w)}{T - T_w}\right)\right)
$$

初期 Adam 统计量不稳（$\hat v$ 噪声大），warmup 避免过早大步长；后期小学习率精修。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `SGD` — SGD
- `Momentum` — Momentum
- `Adam` — Adam
- `AdamW` — AdamW
- `cosine_schedule` — cosine_schedule

- 接口骨架：[`../../code/lab07-optimizers/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab07-optimizers/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab07-optimizers/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab07-optimizers/selftest.py)
- 实验入口：`cd code/lab07-optimizers && uv sync && uv run run_lab07.py`


