---
title: "Lab 11：Residual Connection"
lab: 11
stage: 2
---

对应大纲实验 11。依赖 Lab 09。

## 0. 理论问题

1. 深度增加时，为什么普通网络的梯度会退化？
2. 残差连接在反向传播中改变了什么？
3. 残差网络的「恒等映射」思想为什么对任意深度有效？

## 1. 数学基础

### 1.1 普通深网的梯度衰减

$L$ 层复合 $h_L = f_L \circ \cdots \circ f_1(h_0)$，链式法则：

$$
\frac{\partial h_L}{\partial h_0} = \prod_{l=1}^{L} J_l, \qquad
\left\| \frac{\partial h_L}{\partial h_0} \right\| \le \prod_l \|J_l\| \le \left( \max_l \|J_l\| \right)^L
$$

若每层 Jacobian 谱范数略小于 1（例如 0.9），20 层后梯度 $\sim 0.9^{20} \approx 0.12$，50 层后 $\sim 0.005$：浅层梯度消失，优化停滞。若略大于 1 则指数爆炸。乘性结构是根源。

### 1.2 残差块的梯度高速公路

残差块 $h_{l+1} = h_l + F(h_l)$：

$$
\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F}{\partial h_l}
$$

跨 $L$ 个残差块：

$$
\frac{\partial h_L}{\partial h_0} = \prod_l \left( I + J_{F,l} \right) = I + \sum_l J_{F,l} + \sum_{l<m} J_{F,m} J_{F,l} + \cdots
$$

展开式中恒等项 $I$ 永远存在：任何一条「纯恒等路径」的梯度贡献为 1，不随深度衰减；$F$ 的小扰动项只做加性修正。梯度下界不再乘性缩小——这就是 highway 的含义：即使 $F$ 学不到东西，梯度也能无衰减地穿过。

<iframe src="resources/figures/grad-highway.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="残差连接的梯度高速公路"></iframe>

<p align="center"><b>图 11.1</b>：首层梯度范数 vs 深度</p>

<iframe src="resources/figures/residual-block.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="残差块：学习对恒等的修正"></iframe>

<p align="center"><b>图 11.2</b>：y = x + F(x)</p>

### 1.3 与初始化的配合

若 $F$ 最后一层初始化为零（常见做法），初始网络是严格恒等映射 $h_L = h_0$，训练从「什么都不做」出发逐步增加残差修正。对比普通网络：再好的初始化也无法让 50 层复合恒等于恒等（误差逐层累积）。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `PlainConvBlock` — conv-bn-relu-conv-bn + (可选下采样)
- `ResidualBlock` — 同上 + 跳跃连接；下采样时 1x1 conv 投影对齐维度
- `make_plain_cnn` — 参数量与 plain 对齐（±15%）
- `make_res_cnn` — 参数量与 plain 对齐（±15%）

- 接口骨架：[`../../code/lab11-residual/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab11-residual/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab11-residual/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab11-residual/selftest.py)
- 实验入口：`cd code/lab11-residual && uv sync && uv run run_lab11.py`


