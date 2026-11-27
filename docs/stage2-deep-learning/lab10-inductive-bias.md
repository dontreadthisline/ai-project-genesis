---
title: "Lab 10：卷积结构的归纳偏置"
lab: 10
stage: 2
---

对应大纲实验 10。依赖 Lab 09。

## 0. 理论问题

1. 打乱像素位置后，MLP 与 CNN 的性能变化为什么不对称？
2. 「假设空间更大」与「样本效率更高」之间的权衡如何量化？
3. 等变性是对数据变换的先验：当变换不满足时先验变成负担吗？

## 1. 数学基础

### 1.1 置换等价的重参数化

设固定置换矩阵 $\Pi$，打乱后输入 $x' = \Pi x$。MLP 第一层：$W' x' = W' \Pi x$，取 $W' = W \Pi^{-1}$ 则与原网络在原数据上完全等价。**置换对 MLP 是无害的重参数化**：假设空间不变，只是求解难度可能变化（打破空间局部性会让优化更难，但表达能力不减）。

CNN 则不同：卷积核在 $x'$ 上的局部窗口对应原图的随机散点，局部相关性先验彻底失效；且无法用重参数化恢复（卷积结构本身固定了权重共享模式）。预期：CNN 性能大幅下降，MLP 几乎不变（或略降）。

### 1.2 归纳偏置的偏置-方差语言

模型族 $\mathcal H_{\text{CNN}} \subset$（局部函数类），$\mathcal H_{\text{MLP}}$ 更大。误差分解：

$$
\mathbb E[(y - \hat f)^2] = \underbrace{\text{近似误差}(\mathcal H)}_{\text{先验对：CNN 更低}} + \text{估计误差}(\mathcal H, n) + \text{噪声}
$$

数据少时估计误差主导，小而对的假设类（CNN）赢；数据多时近似误差主导，差距消失甚至反转（若真实映射确非局部）。实验 10 的样本量扫描就是这条曲线的直接观测。

### 1.3 kernel size 与深度的交互

大 kernel 单层感受野大但参数 $K^2$ 增长；小 kernel 堆叠以线性参数量换指数感受野（$(K-1)L$ vs $K^L$ 感受野覆盖）。打乱像素后这一交换完全失效——感受野的「空间连续」前提不存在了。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `make_permutation` — 固定置换，训练/测试共用
- `permute_dataset` — X (N, 28, 28) -> 打乱后 reshape 回

- 接口骨架：[`../../code/lab10-inductive-bias/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab10-inductive-bias/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab10-inductive-bias/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab10-inductive-bias/selftest.py)
- 实验入口：`cd code/lab10-inductive-bias && uv sync && uv run run_lab10.py`


