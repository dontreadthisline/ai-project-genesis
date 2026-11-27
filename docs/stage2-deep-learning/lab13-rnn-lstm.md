---
title: "Lab 13：RNN 与 LSTM 的长期依赖问题"
lab: 13
stage: 2
---

对应大纲实验 13。

## 0. 理论问题

1. BPTT 中梯度为什么会随时间步指数衰减或爆炸？
2. LSTM 的门控与加性细胞状态更新如何缓解这个问题？
3. 「缓解」是否等于「解决」？边界在哪里？

## 1. 数学基础

### 1.1 RNN 与 BPTT

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b), \qquad y_t = W_o h_t
$$

沿时间反向传播（BPTT），损失对 $h_0$ 的梯度：

$$
\frac{\partial L}{\partial h_0} = \prod_{t=1}^{T} W_h^T \mathrm{diag}\!\left(1 - h_t^2\right) \cdot \frac{\partial L}{\partial h_T}
$$

矩阵连乘的范数上界 $\le \prod_t \|W_h\| \cdot \|\mathrm{diag}(1-h_t^2)\|$。tanh 饱和时对角项趋 0，双重压缩：梯度对 $T$ 呈指数缩放。$W_h$ 最大奇异值 $< 1$ 时消失、$> 1$ 时爆炸——乘性结构使「恰好在 1 附近」在深时间维上几乎不可能稳定保持。

<iframe src="resources/figures/bptt-decay.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="BPTT 的指数衰减"></iframe>

<p align="center"><b>图 13.1</b>：梯度范数 vs 序列长度</p>

### 1.2 LSTM

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{遗忘门} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{输入门} \\
\tilde c_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

关键在细胞状态的加性更新：$c_t = f_t \odot c_{t-1} + (\cdot)$，

$$
\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t) + \cdots(\text{对 } c_{t-1} \text{ 的间接路径})
$$

若遗忘门 $f_t \approx 1$，梯度近似无衰减地流过任意时间步（记忆被选择性地保留）；每个维度有独立的、可学习的「保留率」$f_t$，而不是所有维度共享同一个 $W_h$ 谱半径。梯度爆炸仍可能（无裁剪时），消失被结构性缓解。

<iframe src="resources/figures/lstm-cell.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="LSTM：加性细胞状态"></iframe>

<p align="center"><b>图 13.2</b>：细胞状态的加性更新</p>

### 1.3 梯度裁剪

爆炸的标准处理：$\tilde g = g \cdot \min(1, \tau / \|g\|)$。只对超过阈值的方向缩放，不改变方向。

## 2. 实现任务

任务：加法问题（经典 long-term dependency 基准）。序列长 $T$，两个位置 $i, j$ 的输入附加标志 1，其余为 0，第一通道为均匀噪声，目标 = 两标志位置的噪声之和。以及复制记忆任务（可选）。

实现清单（签名与 TODO 见骨架源码）：

- `VanillaRNN` — h_t = tanh(W_h h + W_x x)
- `LSTM` — 1.2 六个方程
- `bptt_grad_norm` — 返回 ∂L/∂h_0 的范数

- 接口骨架：[`../../code/lab13-rnn-lstm/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab13-rnn-lstm/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab13-rnn-lstm/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab13-rnn-lstm/selftest.py)
- 实验入口：`cd code/lab13-rnn-lstm && uv sync && uv run run_lab13.py`


