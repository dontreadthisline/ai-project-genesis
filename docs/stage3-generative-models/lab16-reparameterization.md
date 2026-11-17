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

### 1.2 方差对比的直觉

Score function 的估计量 $f(z)\nabla\log q$ 把 $f$ 的值当作随机乘子：$f$ 的绝对幅度直接进入方差（baseline 技术是 Lab 28 的主题，缓解但不消除）。Pathwise 只用 $\nabla f$ 的局部信息，信号通常小一两个数量级。具体差距取决于 $f$：本实验用数值回答。

### 1.3 对照用例的解析真值

取 $f(z) = \sin(z)$，$z \sim \mathcal N(\mu, \sigma^2)$：

$$
\mathbb E[f] = e^{-\sigma^2/2}\sin\mu, \qquad
\frac{\partial}{\partial \mu}\mathbb E[f] = e^{-\sigma^2/2}\cos\mu
$$

解析梯度存在，两种估计可同时对照真值校验无偏性。

## 2. 实现任务

```python
def sf_grad(f, mu, sigma, n, rng): ...
    # (f(z) - b) * (z-mu)/sigma^2 对 mu 的均值；b 为可选 baseline（默认 0）
def pathwise_grad(f, mu, sigma, n, rng): ...
    # eps ~ N(0,1), z = mu + sigma*eps, 数值梯度 df/dz (中心差分) * 1
```

自测：

```python
# T1: 无偏性：两种估计对 mu 的梯度均值随 n=1e6 收敛到解析值（3 个标准误差内）
# T2: 方差比：n=1 时重复 5000 次的样本方差比 sf/pathwise（记录数量级）
# T3: baseline=b 的 sf：E[b * ∇log q] = 0（无偏不变，方差改变）
# T4: sigma 的梯度同构验证（解析 dE/dσ 存在时 pathwise 成立）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 估计方法 × n $\{1, 10, 10^3\}$ | f=sin，固定 $(\mu,\sigma)$ | 估计量分布直方图、方差 |
| E2 | baseline $b \in \{0, \mathbb E[f]\text{ 的粗估}\}$ | sf 方法 | 方差缩减比例 |
| E3 | $f$ 的幅度缩放 $\{1\times, 10\times\}$ | — | sf 方差 ∝ f²，pathwise 不变 |
| E4 | 在 Lab 15 的 VAE 上换用 sf 估计 | — | 训练可行性对照（预期发散或极慢） |

## 4. 预期图表

- 图 1：六联直方图（方法 × n）：sf 的分布宽且重尾；pathwise 集中在真值附近。
- 图 2：方差比柱状图：典型比值 $10^1 \sim 10^2$。
- 图 3：E3 双对数：sf 方差随 f 幅度平方增长，pathwise 不变——两种估计的「尺度敏感度」本质差异。
- 图 4：E4 loss 曲线：sf-VAE 无法收敛到可比质量。

## 5. 观测指标

`var_ratio_by_n`、`var_ratio_by_scale`、`baseline_reduction`。

## 6. 结果分析

1. 用 1.1 的两个公式解释 E3 中尺度敏感度差异的来源。
2. VAE 的重构项 $f = \log p(x|z)$ 是一个「大数值网络输出」，这对 E4 的结论意味着什么？
3. 离散潜变量（如类别采样）为什么必须用 score function？（gumbel-softmax 是近似 pathwise 的折中，此处只需说明方向。）

## 7. 验收清单

- [ ] T1-T4 通过；能白板推导两种估计的恒等式
- [ ] 能定量说明方差差异数量级及来源
- [ ] 理解 Lab 28（REINFORCE）必须走 score function 路线、Lab 15（VAE）走 pathwise 的原因
