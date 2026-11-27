---
title: "Lab 08：Batch Size 对梯度估计的影响"
lab: 8
stage: 2
---

对应大纲实验 8。依赖 Lab 06/07。这是 Lab 03 Monte Carlo 在优化中的直接复用。

## 0. 理论问题

1. Mini-batch 梯度相对全量梯度是偏差还是噪声问题？
2. 梯度噪声如何随 batch size 缩放？它与「大 batch 泛化差」有什么关系？
3. 相同 epoch 数下，大 batch 与小 batch 谁的 wall-clock 效率高？

## 1. 数学基础

### 1.1 梯度即 Monte Carlo 估计

经验风险 $L(\theta) = \frac1N \sum_i \ell_i(\theta)$，全量梯度 $g^* = \nabla L$。batch $B$（大小 $b$）梯度：

$$
\hat g_B = \frac1b \sum_{i \in B} \nabla \ell_i(\theta)
$$

是 $g^* = \mathbb E_B[\hat g_B]$ 的无偏估计（均匀采样），方差（Lab 03 §1.2 的直接应用）：

$$
\mathrm{Var}[\hat g_B] = \frac{\sigma^2}{b}, \qquad \sigma^2 = \mathrm{Var}_i[\nabla \ell_i(\theta)]
$$

梯度噪声是方差问题而非偏差问题；标准误差按 $1/\sqrt b$ 缩减。

### 1.2 梯度信噪比与有效学习率

SGD 更新 $\Delta\theta = -\eta \hat g$ 的噪声协方差为 $\eta^2 \sigma^2 / b$。若把 batch 扩大 $k$ 倍并保持噪声不变，可将 $\eta$ 放大 $\sqrt k$；线性缩放规则（经验）则取 $\eta \propto b$ 并配 warmup。噪声尺度的一种度量：

$$
\mathrm{SNR}(B) = \frac{\|g^*\|_2}{\|\hat g_B - g^*\|_2}
$$

随 $b$ 以 $\sqrt b$ 增长。噪声本身有隐式正则化作用（flat minima 假说）：大 batch + 大学习率补偿不足时常见泛化下降。

### 1.3 相似度度量

逐 batch 与全量梯度的余弦相似度：

$$
\cos(B) = \frac{\hat g_B \cdot g^*}{\|\hat g_B\| \|g^*\|}
$$

衡量「这一步方向对不对」。经验上 $\cos > 0.9$ 后继续增大 batch 收益递减。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `full_batch_grad` — 全数据集梯度（参考真值）
- `batch_grad` — 返回各 batch size 的方差、SNR、cos
- `grad_stats` — 返回各 batch size 的方差、SNR、cos

- 接口骨架：[`../../code/lab08-batch-size/skeleton.py`](../../code/lab08-batch-size/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab08-batch-size/selftest.py`](../../code/lab08-batch-size/selftest.py)
- 实验入口：`cd code/lab08-batch-size && uv sync && uv run run_lab08.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $b \in \{1, 8, 64, 512, 4096\}$ | 同一 $\theta$（训练中后期快照） | 方差、SNR、余弦相似度 |
| E2 | $b \in \{32, 256, 2048\}$，lr 线性缩放 | 同 epoch 数 | test acc、train/val gap |
| E3 | $b \in \{32, 256, 2048\}$，lr 固定 | — | 达到 95% acc 的 wall-clock 与步数 |
| E4 | 训练早期 vs 中期 vs 后期的 $\cos(B)$ | 固定 b=64 | 噪声随收敛的变化 |

## 4. 预期图表

- 图 1：双对数图，方差 vs b（斜率 $-1$），SNR vs b（斜率 $+1/2$），叠加理论线。
- 图 2：E2 三条 acc 曲线：大 batch + 线性缩放在小数据集上差距不大，但小 batch 的 val 曲线更平滑。
- 图 3：E3 的 wall-clock 柱状图：大 batch 每 epoch 快（并行），但达到同精度所需 epoch 更多，总时间未必占优。
- 图 4：$\cos$ 随训练进程上升（梯度趋于一致）或下降（接近极小值、信噪比变化），如实报告。

## 5. 观测指标

`var_by_b`、`snr_by_b`、`cos_by_b`、`acc_and_time_by_b`。

## 6. 结果分析

1. 用 Lab 03 的公式写出 batch 梯度标准误差，与本实验数值对照。
2. E2/E3 中大 batch 的「并行收益」与「噪声损失」如何权衡？在本实验（MNIST，60k 样本）拐点在哪？
3. 固定计算预算（总样本访问数相同）时，小 batch 多步 vs 大 batch 少步，哪个最终 test acc 高？为什么？

## 7. 验收清单

- [ ] T1-T4 通过；能写出 $\mathrm{Var}[\hat g] = \sigma^2/b$ 并说明它与 Lab 03 的同构性
- [ ] 图 1 两条缩放律拟合斜率分别在 $[-1.1,-0.9]$ 与 $[0.4,0.6]$
- [ ] 能解释梯度噪声作为一种隐式正则的假说及本实验的观察是否支持
