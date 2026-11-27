---
title: "Lab 04：概率分布度量"
subtitle: "Entropy / Cross Entropy / KL / Brier Score"
lab: 4
---

对应大纲实验 4。依赖：MLX。产出：`run_lab04.py`。本实验是所有分类损失（实验 6）、KL 正则（实验 15/30/39）的数学前置。

## 0. 理论问题

1. Entropy、Cross Entropy、KL Divergence 各自度量什么？三者有什么恒等关系？
2. 为什么 KL 非负、不对称？两个方向的不对称各自偏向什么行为？
3. 为什么深度学习分类用 Cross Entropy 而不是 Brier Score？两者在什么意义下等价、什么意义下不等价？

约定：本实验全部用自然对数（单位 nats）。换底只差常数因子。

## 1. 数学基础

### 1.1 三个度量的定义

对离散分布 $P, Q$（同一支撑上）：

$$
H(P) = -\sum_x P(x) \log P(x)
$$

$$
H(P, Q) = -\sum_x P(x) \log Q(x)
$$

$$
D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

$H(P)$ 是 $P$ 自身的不确定度；$H(P,Q)$ 是「用 $Q$ 编码来自 $P$ 的样本所需的平均码长」；KL 是多付出的代价。三者的恒等关系（推导只用了拆项）：

$$
H(P, Q) = -\sum_x P \log Q = \underbrace{-\sum_x P \log P}_{H(P)} + \sum_x P \log \frac{P}{Q} = H(P) + D_{\mathrm{KL}}(P \| Q)
$$

推论：固定 $P$ 时，最小化交叉熵等价于最小化 KL（$H(P)$ 是常数）。这就是「语言模型训练 = 最小化与数据分布的 KL」的来源。

<iframe src="resources/figures/h-ce-kl.html" style="width:100%;height:394px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="交叉熵 = 熵 + KL"></iframe>

<p align="center"><b>图 04.1</b>：三曲线: 最小值在 Q=P 处</p>

### 1.2 KL 非负（Jensen 不等式推导）

设 $f(x) = Q(x)/P(x)$ 在 $\mathrm{supp}(P)$ 上取值。$\log$ 是凹函数，Jensen 不等式给出 $\mathbb E[\log f] \le \log \mathbb E[f]$：

$$
-D_{\mathrm{KL}}(P \| Q) = \sum_x P(x) \log \frac{Q(x)}{P(x)} \le \log \sum_x P(x) \frac{Q(x)}{P(x)} = \log \sum_x Q(x) = \log 1 = 0
$$

故 $D_{\mathrm{KL}} \ge 0$，等号当且仅当 $Q = P$（Jensen 取等条件：$f$ 在 supp 上为常数）。注意：KL 不是距离，不满足对称性和三角不等式。

### 1.3 不对称的两个方向

固定 $P = (0.5, 0.5)$，$Q = (0.99, 0.01)$（nats）：

$$
D_{\mathrm{KL}}(P \| Q) = 0.5 \ln\frac{0.5}{0.99} + 0.5 \ln\frac{0.5}{0.01} \approx 1.614, \qquad
D_{\mathrm{KL}}(Q \| P) \approx 0.637
$$

行为差异（这一差异将直接决定实验 39 DPO 用哪个方向）：

- 前向 $D_{\mathrm{KL}}(P\|Q)$：只要 $P(x) > 0$，则 $Q(x) \to 0$ 会把 KL 推到无穷。最小化它迫使 $Q$ 覆盖 $P$ 的所有模式：mode-covering（质量铺开）。
- 反向 $D_{\mathrm{KL}}(Q\|P)$：在 $P(x) = 0$ 处 $Q$ 必须为 0，但 $P$ 小的地方 $Q$ 可以更小。最小化它允许 $Q$ 集中在 $P$ 的单个模式：mode-seeking（质量集中）。VAE 用前向，典型 GAN 的最优判别器对应 JS，扩散模型变分界也含前向结构，RLHF/DPO 中 policy 相对 reference 的 KL 用 $D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{ref}})$。

<iframe src="resources/figures/kl-asymmetry.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="KL 的不对称"></iframe>

<p align="center"><b>图 04.2</b>：双向 KL: 两个不同的量</p>

### 1.4 分类损失：Cross Entropy vs Brier Score

多分类，真实类别 $y$，模型输出概率向量 $q$。两种损失：

$$
\mathcal L_{\mathrm{CE}}(q, y) = -\log q_y, \qquad
\mathcal L_{\mathrm{Brier}}(q, y) = \sum_{k=1}^{K} \left( q_k - \mathbb 1[k = y] \right)^2
$$

两者都是 proper scoring rule：当样本真实条件概率为 $p$ 时，期望损失 $\mathbb E_y[\mathcal L(q, y)]$ 在 $q = p$ 处取最小（CE 的期望是 $H(p, q)$，由 1.1/1.2 在 $q = p$ 最小；Brier 可直接对 $q$ 求导验证）。差异在对错误自信的惩罚强度：

- CE 的梯度 $\partial(-\log q_y)/\partial q_y = -1/q_y$：$q_y \to 0$ 时梯度无界，错得离谱时惩罚指数级（对数尺度）增长。
- Brier 的梯度 $-2(1 - q_y)$：有界，$q_y \to 0$ 时梯度饱和于 2。

数值感受：$q_y = 10^{-6}$ 时 $\mathcal L_{\mathrm{CE}} \approx 13.8$ nats，而 Brier $\approx 1$。

联系：最小化训练集平均 CE $= \min \frac1N \sum_i -\log q_\theta(y_i \mid x_i)$ 恰好是最小化模型分布与经验分布 $\hat P$ 的交叉熵 $H(\hat P, Q_\theta)$，由 1.1 即等价于最小化 $D_{\mathrm{KL}}(\hat P \| Q_\theta)$ —— 这就是「训练即 KL 最小化」的严格表述，Lab 05 将从 MLE 角度再次得到同一结论。

### 1.5 连续分布的 KL（仅作了解）

高斯对高斯有闭式解（后续 VAE/扩散直接使用）：

$$
D_{\mathrm{KL}}\big( \mathcal N(\mu_1, \sigma_1^2) \,\|\, \mathcal N(\mu_2, \sigma_2^2) \big)
= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac12
$$

本实验用离散格点近似验证此式（把两条密度在 $[-8, 8]$ 上以 $\Delta = 0.01$ 离散化归一后计算离散 KL）。

## 2. 实现任务

在 `metrics.py` 中：

实现清单（模块与接口见源码）：

- `entropy` — H(P)，nats。输入为概率向量（mx.array）。约定 0*log0 = 0。
- `cross_entropy` — H(P, Q)。当 P(x)>0 而 Q(x)=0 时应返回 inf（用 mx.isinf 可测；
- `kl_divergence` — D_KL(P||Q)。要求实现方式为 sum(p * (log p - log q)) 并正确处理 p=0 的项。
- `brier_score` — 多分类 Brier：sum_k (q_k - 1[k==y])^2。

- 模块源码：[`code/lab04-entropy-kl/`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab04-entropy-kl/)
- 自测断言（红灯标准）：`run_lab04.py` 的 `check_implementation()`
- 实验入口：`cd code/lab04-entropy-kl && uv run run_lab04.py`


## 3. 概念接口预告

- 实验 6（MNIST 分类）：loss 曲线 = 本实验的 CE，逐 batch 的 NLL 均值可直接对照。
- 实验 15（VAE）：KL 项 = $D_{\mathrm{KL}}(q(z|x) \| p(z))$，两个高斯的闭式解即 1.5。
- 实验 30（PPO）：KL 惩罚 = $D_{\mathrm{KL}}(\pi_{\mathrm{new}} \| \pi_{\mathrm{old}})$。
- 实验 39（DPO）：隐式奖励由 policy 与 reference 的对数比构成，其期望即反向 KL 的控制项。
