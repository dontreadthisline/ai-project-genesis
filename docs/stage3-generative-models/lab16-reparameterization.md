---
title: "Lab 16：Reparameterization 与梯度传播"
lab: 16
stage: 3
---

对应大纲实验 16。依赖 Lab 03（MC 估计与方差）、Lab 15。

## 0. 理论问题

1. 为什么 $z \sim q_\phi$ 直接采样会切断梯度？数学上缺了哪一步？
2. Pathwise（重参数化）与 score function（REINFORCE）两种梯度估计各自的方差特性？
3. 什么时候只能用 score function？

## 1. 数学基础

### 1.1 两种梯度估计的推导

目标 $\nabla_\phi \mathbb E_{z \sim q_\phi}[f(z)]$。采样操作不可微，需恒等变形。

**Score function（REINFORCE）**：log-derivative 恒等式 $\nabla_\phi q_\phi(z) = q_\phi(z)\, \nabla_\phi \log q_\phi(z)$：

$$
\nabla_\phi \mathbb E_{q_\phi}[f] = \mathbb E_{q_\phi}\left[ f(z)\, \nabla_\phi \log q_\phi(z) \right]
$$

单样本 MC 估计即可。对分布族几乎无要求（离散/不可重参数化均可）。

**Pathwise（重参数化）**：若 $z = g_\phi(\epsilon)$、$\epsilon \sim p(\epsilon)$ 与 $\phi$ 无关：

$$
\nabla_\phi \mathbb E_{q_\phi}[f] = \mathbb E_{\epsilon}\left[ \nabla_z f(z)\, \nabla_\phi g_\phi(\epsilon) \right]
$$

梯度穿过 $f$ 的内部——要求 $f$ 可微且 $q_\phi$ 可重参数化。

<iframe src="resources/figures/two-gradients.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="两条梯度路径"></iframe>

<p align="center"><b>图 16.1</b>：pathwise vs score function</p>

### 1.2 方差对比的直觉

Score function 的估计量 $f(z)\nabla\log q$ 把 $f$ 的值当作随机乘子：$f$ 的绝对幅度直接进入方差（baseline 技术是 Lab 28 的主题，缓解但不消除）。Pathwise 只用 $\nabla f$ 的局部信息，信号通常小一两个数量级。具体差距取决于 $f$：本实验用数值回答。

<iframe src="resources/figures/variance-ratio.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="两种估计的方差差"></iframe>

<p align="center"><b>图 16.2</b>：估计量方差 vs f 的幅度</p>

### 1.3 对照用例的解析真值

取 $f(z) = \sin(z)$，$z \sim \mathcal N(\mu, \sigma^2)$：

$$
\mathbb E[f] = e^{-\sigma^2/2}\sin\mu, \qquad
\frac{\partial}{\partial \mu}\mathbb E[f] = e^{-\sigma^2/2}\cos\mu
$$

解析梯度存在，两种估计可同时对照真值校验无偏性。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `sf_grad` — (f(z) - b) * (z-mu)/sigma^2 对 mu 的均值；b 为可选 baseline（默认 0）
- `pathwise_grad` — eps ~ N(0,1), z = mu + sigma*eps, 数值梯度 df/dz (中心差分) * 1

- 接口骨架：[`../../code/lab16-reparameterization/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab16-reparameterization/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab16-reparameterization/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab16-reparameterization/selftest.py)
- 实验入口：`cd code/lab16-reparameterization && uv sync && uv run run_lab16.py`


