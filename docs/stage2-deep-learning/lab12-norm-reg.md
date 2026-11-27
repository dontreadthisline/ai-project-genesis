---
title: "Lab 12：归一化与正则化对照"
lab: 12
stage: 2
---

对应大纲实验 12。依赖 Lab 06/07/09。

## 0. 理论问题

1. BatchNorm 在训练和推理时的行为为什么不同？这个差异有什么代价？
2. LayerNorm 与 BatchNorm 的归一化轴为什么不同？各自适配什么场景？
3. Dropout 为什么等价于一种集成？它与 weight decay 的正则机制有何区别？

## 1. 数学基础

### 1.1 BatchNorm

对 mini-batch 内每个特征维 $j$：

$$
\hat x_{ij} = \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad
y_{ij} = \gamma_j \hat x_{ij} + \beta_j
$$

$\mu_j, \sigma_j^2$ 是 batch 统计；推理时用训练期的指数滑动平均 $\bar\mu, \bar\sigma$（running stats，momentum $m$：$\bar\mu \leftarrow (1-m)\bar\mu + m\mu_{batch}$）。训练/推理不一致是 BN 的固有代价；batch 内样本相互依赖（同一 batch 的归一化互相影响），大 batch 或序列长度可变时问题加剧。

### 1.2 LayerNorm

对每个样本的所有特征归一化：

$$
\hat x_{i} = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \qquad y_i = \gamma \odot \hat x_i + \beta
$$

统计量沿特征轴而非 batch 轴，逐样本独立：无 batch 依赖、推理与训练一致。代价是没有跨样本的统计共享。Transformer（Lab 14、第五阶段）一律用 LN：序列长度可变 + 自回归推理逐步生成，BN 的 batch/时间统计都不可行。

### 1.3 Dropout（inverted）

训练时以概率 $p$ 置零并缩放：

$$
\tilde h_i = \frac{m_i h_i}{1-p}, \qquad m_i \sim \mathrm{Bernoulli}(1-p)
$$

期望不变 $\mathbb E[\tilde h] = h$（推理时无需缩放）。集成视角：$2^n$ 个 mask 对应 $2^n$ 个子网络共享参数，每步训练随机采样一个子网；推理是权重比例近似的隐式集成。

### 1.4 Weight Decay

L2 惩罚 $\frac\lambda2\|\theta\|^2$ 的梯度 $\lambda\theta$ 是持续向原点的收缩力。MAP 视角（Lab 05 §1.6）：等价于 $\theta \sim \mathcal N(0, 1/\lambda)$ 先验，把解拉向小范数区域。与 Dropout 的差别：decay 作用于参数空间（收缩权重），dropout 作用于激活空间（随机遮蔽）；两者常叠加使用。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `BatchNorm` — train/eval 模式，running stats
- `LayerNorm` — 逐样本
- `dropout` — inverted dropout

- 接口骨架：[`../../code/lab12-norm-reg/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab12-norm-reg/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab12-norm-reg/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab12-norm-reg/selftest.py)
- 实验入口：`cd code/lab12-norm-reg && uv sync && uv run run_lab12.py`


