---
title: "Lab 05：最大似然估计"
lab: 5
---

对应大纲实验 5。依赖：Lab 01（高斯密度）、Lab 04（交叉熵视角）。产出：`run_lab05.py`。本实验把第一阶段的四条线（采样、期望、信息度量、参数估计）收拢成一个结论：MLE = 最小化经验分布与模型分布的交叉熵 = 最小化 KL。这是通往第二阶段「训练神经网络」的桥梁。

## 0. 理论问题

1. 「从数据反推参数」如何形式化？最大似然的解有什么性质？
2. 高斯 MLE 的闭式解是什么？$\sigma^2$ 的 MLE 为什么有偏？偏差多大？
3. 估计量本身的分布（抽样分布）随样本量如何变化？MAP 与 MLE 差在哪？

## 1. 数学基础

### 1.1 似然函数与 MLE

i.i.d. 数据 $x_1, \dots, x_n \sim p(x \mid \theta)$。似然是「把参数当变量、数据当固定量」的联合密度：

$$
L(\theta) = \prod_{i=1}^n p(x_i \mid \theta), \qquad
\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta), \qquad
\hat\theta_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)
$$

取对数的理由：单调不变性保证最优解不变，且把乘积变求和（数值稳定、可分解到样本、导数好算）。

### 1.2 高斯 MLE 闭式解（完整推导）

对 $x_i \sim \mathcal N(\mu, \sigma^2)$，去掉与参数无关的项后：

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log \sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

对 $\mu$ 求导置零：

$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_i (x_i - \mu) = 0
\quad\Longrightarrow\quad
\hat\mu = \frac1n \sum_i x_i = \bar x
$$

对 $\sigma^2$ 求导置零（注意变量是 $\sigma^2$ 不是 $\sigma$）：

$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_i (x_i - \hat\mu)^2 = 0
\quad\Longrightarrow\quad
\hat\sigma^2 = \frac1n \sum_i (x_i - \bar x)^2
$$

二阶条件可验证确为最大值（对数似然在 $(\hat\mu, \hat\sigma^2)$ 处 Hessian 负定），这里略。

### 1.3 $\hat\sigma^2$ 的偏差（完整推导）

关键引理：$\mathbb E\big[ (x_i - \bar x)^2 \big] = \sigma^2 (1 - \tfrac1n)$。推导：记 $\bar x = \mu + \bar\varepsilon$，$\varepsilon_i = x_i - \mu$ i.i.d. 均值 0 方差 $\sigma^2$，则 $x_i - \bar x = \varepsilon_i - \bar\varepsilon$，且

$$
\mathrm{Var}(\varepsilon_i - \bar\varepsilon) = \mathrm{Var}(\varepsilon_i) + \mathrm{Var}(\bar\varepsilon) - 2\,\mathrm{Cov}(\varepsilon_i, \bar\varepsilon) = \sigma^2 + \frac{\sigma^2}{n} - 2\cdot\frac{\sigma^2}{n} = \sigma^2\Big(1 - \frac1n\Big)
$$

（其中 $\mathrm{Cov}(\varepsilon_i, \bar\varepsilon) = \sigma^2/n$，因为 $\bar\varepsilon$ 含 $1/n$ 权重的 $\varepsilon_i$ 自身。）于是

$$
\mathbb E[\hat\sigma^2] = \frac1n \sum_i \mathbb E\big[(x_i - \bar x)^2\big] = \frac{n \cdot \sigma^2(1 - \frac1n)}{n} = \frac{n-1}{n}\sigma^2 < \sigma^2
$$

MLE 系统性低估方差，根源是「先用数据估了 $\bar x$ 再估方差」，残差平方和比围绕真均值的偏小。无偏修正即除以 $n - 1$（Bessel 修正）。$n$ 大时两者差异可忽略，小样本时显著。

### 1.4 抽样分布与 Fisher 信息

$\hat\mu = \bar x$ 是高斯的线性组合，精确分布：

$$
\hat\mu \sim \mathcal N\!\left( \mu,\ \frac{\sigma^2}{n} \right)
$$

与 Fisher 信息对照：单样本对 $\mu$ 的 Fisher 信息 $I(\mu) = \mathbb E[(\partial_\mu \log p)^2] = 1/\sigma^2$，Cramér–Rao 下界为 $1/(nI) = \sigma^2/n$，高斯 MLE 精确达到下界（渐近有效）。更一般地，MLE 渐近正态：

$$
\hat\theta \ \dot\sim\ \mathcal N\!\left( \theta,\ \frac{1}{n I(\theta)} \right)
$$

直觉：样本量 $n$ 时代的估计误差量级是 $1/\sqrt n$，与 Lab 03 的 Monte Carlo 收敛速率完全一致——不是巧合，两者都是「用 $n$ 个独立信息压制随机性」。

### 1.5 MLE = 最小化经验交叉熵 = 最小化 KL（阶段总线索）

定义经验分布 $\hat P = \frac1n \sum_i \delta_{x_i}$。则模型对数据的平均负对数似然：

$$
-\frac1n \ell(\theta) = -\frac1n \sum_i \log Q_\theta(x_i) = \sum_x \hat P(x) \big(-\log Q_\theta(x)\big) = H(\hat P, Q_\theta) = H(\hat P) + D_{\mathrm{KL}}(\hat P \| Q_\theta)
$$

$H(\hat P)$ 与 $\theta$ 无关，故

$$
\hat\theta_{\mathrm{MLE}} = \arg\min_\theta D_{\mathrm{KL}}(\hat P \| Q_\theta)
$$

结论：最小化训练 loss（NLL 均值）、最小化交叉熵、最小化与经验分布的 KL，是同一件事的三种说法。第二阶段起所有 `loss = cross_entropy(...)` 的训练循环，都是这个公式在 $Q_\theta$ 为神经网络时的实例。

### 1.6 MAP（简要）

先验 $\mu \sim \mathcal N(\mu_0, \tau^2)$，$\sigma^2$ 已知。后验最大化等价于最大化 $\ell(\mu) + \log p(\mu)$，对 $\mu$ 求导置零：

$$
\hat\mu_{\mathrm{MAP}} = \frac{n\bar x / \sigma^2 + \mu_0/\tau^2}{n/\sigma^2 + 1/\tau^2}
$$

即样本均值与先验均值的精度加权平均（精度 = 方差之逆）。$n \to \infty$ 时收敛于 MLE；$n$ 小时被拉向先验。这与第二阶段 weight decay（$L2$ 正则 = 高斯先验下的 MAP）直接对接。

## 2. 实现任务

在 `estimators.py` 中：

实现清单（模块与接口见源码）：

- `gaussian_nll` — 负对数似然均值：-mean(log N(x | mu, sigma2))。sigma2 > 0。
- `gaussian_mle` — 闭式解 (mu_hat, sigma2_hat)，sigma2 用 1/n。
- `gaussian_map` — 已知 sigma2 时的 mu_MAP，用 1.6 的精度加权公式。

- 模块源码：[`code/lab05-mle/`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab05-mle/)
- 自测断言（红灯标准）：`run_lab05.py` 的 `check_implementation()`
- 实验入口：`cd code/lab05-mle && uv run run_lab05.py`


## 3. 阶段收束

完成本实验后，第一阶段的核心闭环已经合龙：

$$
\underbrace{\text{采样（Lab 01/03）}}_{\text{数据从分布来}}
\;\longrightarrow\;
\underbrace{\text{条件结构（Lab 02）}}_{\text{变量间的依赖}}
\;\longrightarrow\;
\underbrace{\text{信息度量（Lab 04）}}_{H, \mathrm{CE}, \mathrm{KL}}
\;\longrightarrow\;
\underbrace{\text{参数估计（Lab 05）}}_{\mathrm{MLE} = \min H(\hat P, Q_\theta)}
$$

第二阶段实验 6 的 MNIST 分类器，就是把 $Q_\theta$ 换成 MLP、把交叉熵按类别输出逐样本计算，其余不变。届时回看本讲义 1.5 节即可。
