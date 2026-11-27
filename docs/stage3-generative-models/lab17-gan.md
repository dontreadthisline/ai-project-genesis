---
title: "Lab 17：实现基础 GAN"
lab: 17
stage: 3
---

对应大纲实验 17。数据集：MNIST。依赖 Lab 04（JS = 和的 KL 平均）、Lab 06。

## 0. 理论问题

1. 固定 G 时最优判别器是什么？把最优 D 代入后 G 的目标变成什么？
2. 极小极大博弈为什么天然不稳定？两个网络互相追逐意味着什么？
3. Mode collapse 的机制是什么？如何观测？

## 1. 数学基础

### 1.1 目标函数

$$
\min_G \max_D V(D, G), \qquad
V = \mathbb E_{x \sim p_{data}}[\log D(x)] + \mathbb E_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

### 1.2 最优判别器（推导）

固定 G，对单个 $x$，$D$ 的最优值最大化 $p_{data}(x)\log D + p_g(x)\log(1-D)$。对 $D$ 求导置零：

$$
\frac{p_{data}}{D} - \frac{p_g}{1-D} = 0 \ \Longrightarrow\ D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

<iframe src="resources/figures/d-optimal.html" style="width:100%;height:394px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="最优判别器与均衡"></iframe>

<p align="center"><b>图 17.1</b>：p_data、p_g 与 D*</p>

### 1.3 全局最优：JS 散度（推导）

把 $D^*$ 代回 $V$。记 $a = p_{data}, b = p_g$：

$$
V(G, D^*) = a\log\frac{a}{a+b} + b\log\frac{b}{a+b} + (a + b)\log 2 - (a+b)\log 2 \cdot 1
$$

逐点整理（注意两个期望都是对同一 $x$ 空间积分）：

$$
C(G) = V(D^*, G) = -\log 4 + \mathrm{KL}\left(p_{data} \,\|\, \tfrac{p_{data}+p_g}{2}\right) + \mathrm{KL}\left(p_g \,\|\, \tfrac{p_{data}+p_g}{2}\right) = -\log 4 + 2\,\mathrm{JS}(p_{data} \| p_g)
$$

最小值 $-\log 4$ 当且仅当 $p_g = p_{data}$：博弈的 Nash 均衡即分布匹配。

### 1.4 实践形态与不稳定

原始 $\log(1-D(G(z)))$ 在 G 差时梯度饱和，实践改为最大化 $\log D(G(z))$（非饱和损失）——它不再对应同一 JS 理论。JS 梯度消失问题：不重叠的低维流形间 JS 为常数 $\log 2$，梯度为零，训练信号断裂。Mode collapse：G 发现个别 $z$ 区域的样本能稳定骗过 D，就把多种 $z$ 都映射到同一输出模式——损失看不出异常，只能靠生成多样性观测。

<iframe src="resources/figures/loss-oscillation.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="GAN 训练的拉锯"></iframe>

<p align="center"><b>图 17.2</b>：D/G 损失的动态均衡</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `Generator` — z(64) -> 256 -> 512 -> 784 sigmoid
- `Discriminator` — 784 -> 512 -> 256 -> 1 sigmoid
- `d_loss` — BCE with logits
- `g_loss` — g_loss

- 接口骨架：[`../../code/lab17-gan/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab17-gan/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab17-gan/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab17-gan/selftest.py)
- 实验入口：`cd code/lab17-gan && uv sync && uv run run_lab17.py`


