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

### 1.3 梯度裁剪

爆炸的标准处理：$\tilde g = g \cdot \min(1, \tau / \|g\|)$。只对超过阈值的方向缩放，不改变方向。

## 2. 实现任务

任务：加法问题（经典 long-term dependency 基准）。序列长 $T$，两个位置 $i, j$ 的输入附加标志 1，其余为 0，第一通道为均匀噪声，目标 = 两标志位置的噪声之和。以及复制记忆任务（可选）。

```python
class VanillaRNN:  # h_t = tanh(W_h h + W_x x)
class LSTM:        # 1.2 六个方程
def bptt_grad_norm(model, T, seed): ...   # 返回 ∂L/∂h_0 的范数
```

自测：

```python
# T1: 手算单步：给定 W_h, h，RNN 一步前向与手算一致
# T2: LSTM c_t 更新：f=1, i=0 时 c_t == c_{t-1}（恒等记忆）
# T3: 梯度缩放实证：随机 W_h（谱半径 0.5）下 ||∂L/∂h_0|| 随 T 单调指数下降；
#     LSTM（f 门偏置初始化 +1，靠近 1）下降缓慢（T=50 时比值 > 100 倍）
# T4: 梯度裁剪：norm=10 的梯度裁剪到 5 后范数为 5、方向不变（余弦 = 1）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $T \in \{20, 50, 100, 200\}$ × RNN/LSTM | 隐藏维对齐参数量 | 加法任务 MSE |
| E2 | $T$ 扫描下 $\|\partial L/\partial h_0\|$ | — | 对数图斜率 |
| E3 | LSTM 遗忘门偏置 $\{0, 1, 2\}$ | — | 长序列收敛 |
| E4 | 有无梯度裁剪（RNN，T=100） | — | 训练稳定性（loss 尖峰） |

## 4. 预期图表

- 图 1：MSE vs T：RNN 在 T 增大时停在随机水平平台（≈预测均值 0 的方差），LSTM 保持低误差到 T=100+。
- 图 2：对数-对数图：RNN 梯度范数斜率显著为负（指数消失），LSTM 接近 0。
- 图 3：E3：遗忘门偏置 +1（初始 f≈0.73）优于 0；偏置过大（f≈0.88 起步）短期任务略慢。
- 图 4：无裁剪 RNN 的 loss 曲线出现发散尖峰；裁剪后平滑。

## 5. 观测指标

`mse_by_T`、`grad_norm_by_T`、`best_mse_rnn_vs_lstm`。

## 6. 结果分析

1. 用 1.1 的连乘界估算 RNN 在 $\rho(W_h)=0.5$、$T=100$ 时的梯度量级，与图 2 对照。
2. LSTM 在哪个 $T$ 之后也开始失败？失败模式与 RNN 相同吗？
3. 为什么说 LSTM 是「缓解」：写出它仍会梯度消失的一条具体路径（$f_t$ 被学习为小值时）。

## 7. 验收清单

- [ ] T1-T4 通过；能白板写出 BPTT 连乘式与 LSTM 细胞状态梯度
- [ ] 复现 RNN/LSTM 在加法任务上的 T-性能曲线分化
- [ ] 能解释遗忘门偏置初始化为正的经验做法
