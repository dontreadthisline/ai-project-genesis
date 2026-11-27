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

- 接口骨架：[`../../code/lab12-norm-reg/skeleton.py`](../../code/lab12-norm-reg/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab12-norm-reg/selftest.py`](../../code/lab12-norm-reg/selftest.py)
- 实验入口：`cd code/lab12-norm-reg && uv sync && uv run run_lab12.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | norm：无 / BN / LN（CNN 上） | 训练 30 epoch | 收敛速度、train/val gap |
| E2 | dropout $p \in \{0, 0.3, 0.5\}$ | 有 BN | train/val gap、test acc |
| E3 | weight decay $\lambda \in \{0, 10^{-4}, 10^{-2}\}$ | AdamW | 参数范数、泛化 |
| E4 | 组合消融：BN only / BN+drop / BN+wd / 全部 | — | test acc 与 gap |

## 4. 预期图表

- 图 1：E1 三条 loss 曲线：BN 收敛最快；LN 在 CNN 上也有帮助但略逊（图像特征维统计稳定时 BN 的跨样本信息更有用）。
- 图 2：E2 train/val 双曲线：$p$ 越大 train acc 越低、gap 越小——正则化的可视化。
- 图 3：E3 参数范数随 epoch：decay 越大范数越小且趋于平稳。
- 图 4：E4 柱状图 + gap 标注：MNIST 上正则收益有限（任务简单），如实报告。

## 5. 观测指标

`acc_by_norm`、`gap_by_dropout`、`param_norm_by_wd`、`acc_by_combo`。

## 6. 结果分析

1. E1 中 BN 加速的机制是什么？（内部协变量偏移的解释已被挑战，更可信的是损失曲面平滑化/有效 lr 增大。写出你的观察支持哪种。）
2. E2 中 MNIST 上 dropout 的收益是否显著？什么任务规模下你会预期更大收益？
3. T2 的 train/eval 差异在什么场景会变成真实 bug？（batch=1 微调、分布式训练统计不同步。）

## 7. 验收清单

- [ ] T1-T4 通过；能写出 BN/LN 的公式并说明归一化轴的差别
- [ ] 能解释 inverted dropout 的期望不变性与集成视角
- [ ] 能用 MAP 视角（Lab 05）解释 weight decay
- [ ] 完成组合消融并如实报告 MNIST 上正则收益有限这一事实
