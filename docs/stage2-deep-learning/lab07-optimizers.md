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

- 接口骨架：[`../../code/lab07-optimizers/skeleton.py`](../../code/lab07-optimizers/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab07-optimizers/selftest.py`](../../code/lab07-optimizers/selftest.py)
- 实验入口：`cd code/lab07-optimizers && uv sync && uv run run_lab07.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 优化器：SGD / Momentum / Adam / AdamW | 同一初始化与调度 | loss 曲线、test acc |
| E2 | 学习率 $\{0.1, 0.01, 0.001\} \times$ 优化器 | — | 发散/震荡/过慢分类 |
| E3 | 调度：恒定 / cosine / cosine+warmup | Adam，lr_max 相同 | 最终 test acc |
| E4 | 有无 weight decay（AdamW） | 长训练 50 epoch | train/val gap |

## 4. 预期图表

- 图 1：四条 loss 曲线（对数 y 轴）。Adam 族早期下降最快；调好 lr 的 Momentum 最终精度相当。
- 图 2：E2 的 $4\times3$ 小图矩阵（发散标红）：SGD 容忍大 lr，Adam 小 lr 更稳。
- 图 3：梯度范数与参数范数随 epoch 变化（AdamW 的参数范数缓慢下降 = 衰减生效）。
- 图 4：E3 三条曲线，warmup+cosine 最好（差距小但一致）。

## 5. 观测指标

`best_test_acc_by_optimizer`、`epochs_to_97pct`、`final_param_norm`、`train_val_gap`。

## 6. 结果分析

1. 用 1.1 的展开式解释图 1 中 Momentum 曲线的震荡幅度小于 SGD。
2. E2 中 Adam 在 lr=0.1 时发生什么？（自适应步长 ~lr，对 MNIST-MLP 过大。）
3. E4 中 weight decay 对 train/val gap 的影响是否显著？与 Lab 05 的 MAP 收缩预测一致吗？

## 7. 验收清单

- [ ] T1-T4 通过；能白板推导 Adam 偏差修正因子 $1/(1-\beta^t)$
- [ ] 能解释 Momentum 的方向累积与 Adam 的逐参数步长两种机制的差别
- [ ] 能说出 AdamW 与 Adam+L2 在数学上的差别及后果
- [ ] 至少一组配置 test acc $\geq 98\%$
