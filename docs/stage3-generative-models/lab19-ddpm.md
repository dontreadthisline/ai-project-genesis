---
title: "Lab 19：实现 DDPM"
lab: 19
stage: 3
---

对应大纲实验 19。数据集：MNIST。依赖 Lab 02（高斯条件链）。

## 0. 理论问题

1. 前向过程为什么设计成加噪马尔可夫链？它的边缘分布为什么有闭式解？
2. 训练目标「预测噪声」是怎么从 ELBO 推出来的？
3. 反向采样每步为什么是高斯？

## 1. 数学基础

### 1.1 前向过程与边缘闭式解（推导）

$$
q(x_t \mid x_{t-1}) = \mathcal N\!\left( x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I \right)
$$

关键递推：两个高斯的卷积仍是高斯。设 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod_{s\le t}\alpha_s$：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal N(0, I)
$$

归纳证明：$x_1 = \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1$ 成立；设 $x_{t-1} = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\bar\epsilon$，代入一步转移：

$$
x_t = \sqrt{\alpha_t}\sqrt{\bar\alpha_{t-1}}x_0 + \underbrace{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_{t-1}}\bar\epsilon + \sqrt{1-\alpha_t}\epsilon_t}_{\text{两个独立零均值高斯之和}}
$$

噪声方差相加 $= \alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t) = 1 - \bar\alpha_t$。任意 $t$ 直接采样 $x_t$，训练时无需走完整条链。

### 1.2 反向条件与训练目标

固定 $x_0$ 的反向条件是高斯（推导是对 $q(x_{t-1}|x_t, x_0) \propto q(x_t|x_{t-1})q(x_{t-1}|x_0)$ 配方，用 Lab 02 的精度矩阵技术）：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal N\!\left( \mu, \tilde\beta_t I \right), \quad
\mu = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon \right)
$$

真实反向 $q(x_{t-1}|x_t)$ 边缘掉 $x_0$ 后无闭式，但被高斯近似（$\beta_t$ 小时准确）。参数化 $\epsilon_\theta(x_t, t) \approx \epsilon$：把 1.1 的 $\epsilon = (x_t - \sqrt{\bar\alpha_t}x_0)/\sqrt{1-\bar\alpha_t}$ 代入 $\mu$ 得

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t) \right)
$$

简化 ELBO 后的核心损失（噪声预测 MSE）：

$$
\mathcal L_{simple} = \mathbb E_{t, x_0, \epsilon}\left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t) \|^2 \right]
$$

直觉：网络从带噪图里减出干净图的成分，等价于预测「加进去的噪声」。

### 1.3 采样

从 $x_T \sim \mathcal N(0,I)$ 起，逐步 $x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z$（$\sigma_t^2 = \tilde\beta_t$ 或 $\beta_t$，$t=1$ 时无噪声）。线性 $\beta_t \in [10^{-4}, 0.02]$，$T = 300$（MNIST 小图可用较短链）。

## 2. 实现任务

<div class="theorem algorithm">
<div class="theorem-title">算法 19.1（DDPM 训练与采样）</div>

**训练**：

1. **repeat**　采样 $x_0 \sim p_{data}$，$t \sim \mathrm{U}\{1, \dots, T\}$，$\epsilon \sim \mathcal N(0, I)$；
2. 一步前向 $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$；
3. 梯度步 $\nabla_\theta \| \epsilon - \epsilon_\theta(x_t, t) \|^2$。

**采样**：

1. $x_T \sim \mathcal N(0, I)$；
2. **for** $t = T, \dots, 1$ **：**
   $$x_{t-1} = \tfrac{1}{\sqrt{\alpha_t}}\Big( x_t - \tfrac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t) \Big) + \sigma_t z, \quad z \sim \mathcal N(0, I)$$

</div>

实现清单（签名与 TODO 见骨架源码）：

- `Diffusion` — 线性或余弦
- `EpsNet` — 简化 UNet：conv + 时间嵌入（sinusoidal）

- 接口骨架：[`../../code/lab19-ddpm/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab19-ddpm/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab19-ddpm/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab19-ddpm/selftest.py)
- 实验入口：`cd code/lab19-ddpm && uv sync && uv run run_lab19.py`


