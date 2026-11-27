---
title: "Lab 18：二维 Normalizing Flow"
lab: 18
stage: 3
---

对应大纲实验 18。目标数据：二维高斯混合（两个月亮亦可）。本实验目标是变量替换公式的直觉，不追求复杂架构。

## 0. 理论问题

1. 可逆变换下概率密度如何变换？Jacobian 行列式从哪来？
2. 什么样的变换既可逆又能高效算行列式？
3. 精确似然（对 VAE/GAN 的 ELBO/无密度）意味着什么？

## 1. 数学基础

### 1.1 变量替换公式（推导）

设 $z \sim p_z$，$x = f(z)$，$f$ 可微可逆。概率质量守恒：$z$ 空间小区域 $dz$ 的质量等于 $x$ 空间像区域 $dx$ 的质量：

$$
p_z(z)\, |dz| = p_x(x)\, |dx|, \qquad |dx| = \left| \det \frac{\partial f}{\partial z} \right| |dz|
$$

得 $p_x(x) = p_z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|$。对数似然：

$$
\log p_x(x) = \log p_z(z) - \log\left| \det \frac{\partial f}{\partial z} \right|, \qquad z = f^{-1}(x)
$$

行列式度量体积伸缩：变换在某方向压缩体积，密度必须增大补偿。训练 = 最大化数据点上的 $\log p_x$。

<iframe src="resources/figures/flow-transform.html" style="width:100%;height:394px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="可逆变换的密度传输"></iframe>

<p align="center"><b>图 18.1</b>：圆云 → 双峰（体积补偿）</p>

### 1.2 三角结构的行列式

任意可逆变换的 Jacobian 行列式是 $O(d^3)$（LU 分解）。上/下三角矩阵的行列式 = 对角线乘积，$O(d)$。RealNVP 仿射耦合层：把 $x$ 拆成两半，变换一半、条件于另一半：

$$
h_1 = x_1, \qquad h_2 = x_2 \odot \exp(s(x_1)) + t(x_1)
$$

Jacobian 是块三角：

$$
J = \begin{pmatrix} I & 0 \\ \partial h_2/\partial x_1 & \mathrm{diag}(\exp s) \end{pmatrix}
\ \Longrightarrow\ \det J = \exp\!\left( \sum_j s(x_1)_j \right)
$$

$s, t$ 是任意网络（不需可逆）：逆 $x_2 = (h_2 - t(h_1)) \odot \exp(-s(h_1))$ 显式可写。多层交错变换两半，积累表达能力。

<iframe src="resources/figures/logdet-compensation.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="Jacobian 行列式的补偿"></iframe>

<p align="center"><b>图 18.2</b>：log|det| 随 Σs 的线性关系</p>

### 1.3 采样与密度的对偶

采样：$z \sim \mathcal N(0, I) \to x = f(z)$（前向）。密度：$x \to z = f^{-1}(x)$，上式求值。flow 是唯一「采样便宜 + 密度精确」兼得的范式（VAE 密度只有下界，GAN 无密度）。

## 2. 实现任务

4-6 个耦合层（$s, t$ 为 2 层小 MLP），目标分布：两个高斯混合 $\tfrac12\mathcal N((-2,0), I) + \tfrac12\mathcal N((2,0), I)$。

实现清单（签名与 TODO 见骨架源码）：

- `AffineCoupling` — 实现 forward / inverse / log_det
- `Flow` — 堆叠 coupling，交替 mask 两半

- 接口骨架：[`../../code/lab18-flow/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab18-flow/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab18-flow/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab18-flow/selftest.py)
- 实验入口：`cd code/lab18-flow && uv sync && uv run run_lab18.py`


