---
title: "Lab 06：从零实现 MLP"
lab: 6
stage: 2
---

对应大纲实验 6。数据集：MNIST。产出：`run_lab06.py`。

## 0. 理论问题

1. 前向传播和反向传播在计算上分别做了什么？为什么反向传播比数值微分高效？
2. 交叉熵加 softmax 的梯度为什么有如此简单的形式？
3. 参数初始化的方差为什么影响训练成败？

## 1. 数学基础

### 1.1 线性层与链式法则

线性层 $y = Wx + b$（$W \in \mathbb R^{out \times in}$）。对输入的 Jacobian 是 $W$，对参数的梯度由链式法则给出。反向传播的核心对象是 VJP（向量-Jacobian 积）：

$$
\bar x = W^T \bar y, \qquad \bar W = \bar y\, x^T, \qquad \bar b = \bar y
$$

其中 $\bar{(\cdot)}$ 表示上游梯度。整网反向 = 逆序逐层应用 VJP，代价与前向同阶；数值微分对 $P$ 个参数各需两次前向，代价 $O(P)$ 倍。

### 1.2 Softmax 与交叉熵的合并梯度

softmax：$p_i = e^{z_i} / \sum_j e^{z_j}$，交叉熵 $L = -\log p_y$。推导 $\partial L / \partial z$：

$$
\frac{\partial L}{\partial z_i} = -\frac{\partial \log p_y}{\partial z_i}, \qquad
\frac{\partial p_j}{\partial z_i} = p_j(\delta_{ij} - p_i)
$$

当 $i = y$ 时 $\partial \log p_y/\partial z_y = \frac{1}{p_y} p_y(1-p_y) = 1 - p_y$；当 $i \neq y$ 时 $= -p_i$。合并：

$$
\boxed{\ \frac{\partial L}{\partial z} = p - \mathbf 1_y\ }
$$

梯度就是「预测概率减独热标签」。这是 Lab 04「CE 是分类标准损失」在梯度层面的原因：错误越大概率越大，梯度无饱和区。

### 1.3 数值梯度检验

中心差分 $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$ 误差 $O(h^2)$（泰勒展开相消）。取 $h = 10^{-6}$（float64），解析梯度与数值梯度的相对误差

$$
\mathrm{err} = \frac{\|\nabla_a - \nabla_n\|_2}{\|\nabla_a\|_2 + \|\nabla_n\|_2 + \epsilon}
$$

应小于 $10^{-6}$。这是所有手写反向传播的标准验证手段。

### 1.4 初始化

若线性层输入各分量独立零均值方差 $\sigma_x^2$，则 $y_i = \sum_j W_{ij} x_j$ 的方差为 $in \cdot \mathrm{Var}(W) \cdot \sigma_x^2$。要保持逐层方差稳定（前向不爆炸不消失）需 $\mathrm{Var}(W) = 2/(fan_{in} + fan_{out})$（Xavier，配 tanh）；配合 ReLU 时一半神经元死亡，改用 $\mathrm{Var}(W) = 2/fan_{in}$（He 初始化）。

## 2. 实现任务

结构 $784 \to 256 \to 128 \to 10$，激活 ReLU，输出交叉熵。**用 MLX 实现**（`mx.array` 张量运算），但**不用框架自动微分**：手写 forward/backward，不调用 `mx.grad`。数值梯度检验用 `dtype=mx.float64`（MLX 的 fp64 在 CPU 流水线执行，精度满足 $10^{-6}$ 相对误差）。

实现清单（签名与 TODO 见骨架源码）：

- `Linear` — He 初始化 W, 零初始化 b
- `MLP` — 逐层 Linear + ReLU
- `softmax_cross_entropy` — 数值稳定：logits 减最大值。返回 (loss, dlogits)。dlogits 用 1.2 的闭式解。

- 接口骨架：[`../../code/lab06-mlp-from-scratch/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab06-mlp-from-scratch/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab06-mlp-from-scratch/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab06-mlp-from-scratch/selftest.py)
- 实验入口：`cd code/lab06-mlp-from-scratch && uv sync && uv run run_lab06.py`

实现清单（签名与 TODO 见骨架源码）：

- 接口骨架：[`../../code/lab06-mlp-from-scratch/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab06-mlp-from-scratch/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab06-mlp-from-scratch/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab06-mlp-from-scratch/selftest.py)
- 实验入口：`cd code/lab06-mlp-from-scratch && uv sync && uv run run_lab06.py`


