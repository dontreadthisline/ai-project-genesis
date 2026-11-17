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

```python
class Diffusion:
    def betas_schedule(T): ...          # 线性或余弦
    def q_sample(self, x0, t, eps): ... # 1.1 边缘采样
    def loss(self, x0): ...        # 1.2 simple loss
    def sample(self, n): ...       # 1.3 完整链
class EpsNet: ...                        # 简化 UNet：conv + 时间嵌入（sinusoidal）
```

自测：

```python
# T1: 边缘统计：固定 x0，采样 q(x_t|x0) 1e4 次，均值 ≈ sqrt(abar_t)*x0、
#     方差 ≈ 1-abar_t（容差 3%）
# T2: t=T 时 abar_T ≈ 0（x_T 近似标准正态，与 x0 无关）
# T3: 训练收敛后 loss < 0.1 量级（MNIST 小模型）
# T4: 噪声预测 vs x0 预测等价性：eps_hat 换算 x0_hat = (x_t - sqrt(1-abar)*eps)/sqrt(abar)
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 加噪可视化：$t \in \{0, 50, 150, 300\}$ | 固定 x0 | 噪声淹没信号的进程 |
| E2 | 调度：线性 vs 余弦 | 同预算 | loss、样本质量 |
| E3 | 去噪可视化：采样链中 $x_{300} \to x_0$ 快照 | — | 结构涌现顺序 |
| E4 | $\epsilon$-预测 vs $x_0$-预测参数化 | — | 训练稳定性（小 t 大 t 的损失权重差异） |

## 4. 预期图表

- 图 1：加噪八联图：数字逐渐被噪声吞没，$t=T$ 时纯噪声。
- 图 2：E2 两条 loss 曲线 + 样本对比（余弦调度末端更细腻）。
- 图 3：反向去噪八联图：先全局布局后局部细节。
- 图 4：E4 双曲线：x0 预测在大 $t$ 损失大（要凭空猜结构）、eps 预测各 $t$ 更均匀——这正是 simple loss 选择 eps 的原因。

## 5. 观测指标

`final_loss`、`sample_grid`、`abar_curve`、`loss_by_t`（按 t 分桶的损失）。

## 6. 结果分析

1. 用 1.1 的归纳论证解释「任意 t 直跳采样」为什么对训练至关重要（否则每步都要走链）。
2. `loss_by_t` 呈现什么形状？小 t（噪声少，预测准）与大 t（几乎纯噪声）的损失各说明什么？
3. 图 3 的「先整体后细节」与 $\bar\alpha_t$ 的衰减节奏如何对应？

## 7. 验收清单

- [ ] T1-T4 通过；能白板推导边缘闭式解（归纳步骤）与 $\mu_\theta$ 公式
- [ ] 能解释 eps 参数化与 simple loss 的动机
- [ ] 生成样本可辨认数字（FID 不作要求）
