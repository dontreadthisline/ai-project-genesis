---
title: "Lab 03：Monte Carlo 估计"
lab: 3
---

对应大纲实验 3。依赖：MLX。产出：`run_lab03.py`。本实验是后续策略梯度（实验 28/30/45）、ELBO 采样估计（实验 15）、reward 期望估计（实验 44）的直接数学前置。

## 0. 理论问题

1. 用采样估计 $\mathbb E[f(X)]$ 时，估计误差由什么决定？随样本量如何衰减？
2. 不同被积函数 $f$ 的估计难度差别有多大？什么样的 $f$ 让 Monte Carlo 变得不可行？
3. 重复实验时，估计量自身的分布长什么样？

## 1. 数学基础

### 1.1 估计量及其无偏性

设 $X \sim p$，目标是 $\mu = \mathbb E[f(X)] = \int f(x) p(x)\, dx$。i.i.d. 采样 $X_1, \dots, X_n \sim p$，构造

$$
\hat\mu_n = \frac{1}{n} \sum_{i=1}^n f(X_i)
$$

无偏性由期望的线性立刻得到：

$$
\mathbb E[\hat\mu_n] = \frac{1}{n}\sum_{i=1}^n \mathbb E[f(X_i)] = \frac{1}{n} \cdot n\mu = \mu
$$

无偏只保证「平均而言正确」，单次估计的好坏由方差决定。

### 1.2 方差与标准误差

由样本独立、$\mathrm{Var}[f(X_i)] = \mathrm{Var}[f(X)] \equiv \sigma_f^2$：

$$
\mathrm{Var}[\hat\mu_n] = \frac{1}{n^2} \sum_{i=1}^n \mathrm{Var}[f(X_i)] = \frac{\sigma_f^2}{n}
$$

标准误差（standard error）：

$$
\mathrm{se}(\hat\mu_n) = \frac{\sigma_f}{\sqrt n}
$$

核心结论：误差以 $n^{-1/2}$ 衰减。把精度提高一个数量级，样本量要增加两个数量级。这是所有基于采样的方法（包括 RL 中的策略梯度）的根本成本结构。

实际中 $\sigma_f$ 未知，用样本标准差估计：

$$
\hat\sigma_f^2 = \frac{1}{n-1}\sum_{i=1}^n \left( f(X_i) - \hat\mu_n \right)^2, \qquad \hat{\mathrm{se}} = \hat\sigma_f / \sqrt n
$$

### 1.3 中心极限定理

$$
\frac{\hat\mu_n - \mu}{\sigma_f / \sqrt n} \ \xrightarrow{d}\ \mathcal N(0, 1)
$$

即不管 $f(X)$ 的原始分布多么偏斜，均值估计量在 $n$ 较大时近似正态。这给出了误差区间：$|\hat\mu_n - \mu| \lesssim 2\, \hat{\mathrm{se}}$ 的概率约 95%。实验中要用重复实验验证这一点。

### 1.4 相对误差与稀有事件

估计 $\mu$ 时，真正有意义的刻度是相对误差 $\mathrm{se}(\hat\mu_n) / |\mu|$。对指示函数 $f = \mathbb 1[A]$（估计 $p_A = P(A)$）：

$$
\sigma_f^2 = p_A(1 - p_A), \qquad \frac{\mathrm{se}}{p_A} = \sqrt{\frac{1 - p_A}{n\, p_A}} \approx \frac{1}{\sqrt{n\, p_A}}
$$

若 $p_A = 0.001$、想要 10% 相对误差，需要 $n \ge (1-p_A)/(p_A \cdot 0.1^2) \approx 10^5$：比事件本身罕见一千倍的样本量才能换来一个还不错的估计。稀有事件是朴素 Monte Carlo 的死穴（重要性采样的动机，本课在 RL 阶段再展开）。

### 1.5 二维均匀采样估计 π

在 $[-1,1]^2$ 均匀采样，$f = \mathbb 1[x^2 + y^2 \le 1]$，则 $\mathbb E[f] = \pi/4$。这是「把解析量改写成期望」的最小例子，也是调试采样代码的标准用例。

## 2. 实现任务

在 `estimators.py` 中：

实现清单（模块与接口见源码）：

- `mc_estimate` — f: 向量化函数 mx.array -> mx.array；sampler: callable(n) -> mx.array。

- 模块源码：[`code/lab03-monte-carlo/`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab03-monte-carlo/)
- 自测断言（红灯标准）：`run_lab03.py` 的 `check_implementation()`
- 实验入口：`cd code/lab03-monte-carlo && uv run run_lab03.py`


## 3. 进一步实验（选做）

被积函数与采样分布不匹配的对照：估计 $\mathbb E_{X \sim \mathcal N(0,1)}[e^{X/2}]$（解析解 $e^{1/8}$），再故意改用 $\mathcal N(0, 3)$ 采样但不加权重（错误做法）与加上重要性权重 $w = p(x)/q(x)$ 的正确做法对比。观察权重的方差如何影响估计稳定性。此实验为选做，重要性采样正式进入主线在 RL 阶段（PPO 的 ratio）。
