---
title: "Lab 04：概率分布度量"
subtitle: "Entropy / Cross Entropy / KL / Brier Score"
lab: 4
---

对应大纲实验 4。依赖：NumPy。产出：`experiments/lab04_information_metrics.py`。本实验是所有分类损失（实验 6）、KL 正则（实验 15/30/39）的数学前置。

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

```python
import numpy as np

_EPS = 1e-12

def entropy(p: np.ndarray) -> float:
    """H(P)，nats。输入为概率向量。约定 0*log0 = 0。"""
    ...

def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(P, Q)。当 P(x)>0 而 Q(x)=0 时应返回 inf（不许用 eps 截断掩盖这个行为，
    但实现里可用 eps 防止 log(0) 的 NaN——思考两者区别并写进报告）。"""
    ...

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P||Q)。要求实现方式为 sum(p * (log p - log q)) 并正确处理 p=0 的项。"""
    ...

def brier_score(q: np.ndarray, y: int) -> float:
    """多分类 Brier：sum_k (q_k - 1[k==y])^2。"""
    ...
```

自测：

```python
import numpy as np

# T1: 恒等式 H(P,Q) = H(P) + KL(P||Q) 对随机分布成立
rng = np.random.default_rng(0)
p = rng.dirichlet([1]*5); q = rng.dirichlet([1]*5)
lhs = cross_entropy(p, q); rhs = entropy(p) + kl_divergence(p, q)
assert abs(lhs - rhs) < 1e-10

# T2: KL 非负，且 KL(P||P)=0
assert kl_divergence(p, p) == 0.0
assert kl_divergence(p, q) >= 0.0

# T3: 均匀分布熵 = log K
u = np.ones(5)/5
assert abs(entropy(u) - np.log(5)) < 1e-12

# T4: 确定性分布熵 = 0
assert entropy(np.array([0.0, 1.0, 0.0])) == 0.0

# T5: 支撑违规 -> inf
p_bad = np.array([0.5, 0.5, 0.0]); q_bad = np.array([0.7, 0.0, 0.3])
assert np.isinf(cross_entropy(p_bad, q_bad))

# T6: 手算对照（1.3 的例子）
P = np.array([0.5, 0.5]); Q = np.array([0.99, 0.01])
assert abs(kl_divergence(P, Q) - 1.6144) < 5e-3
assert abs(kl_divergence(Q, P) - 0.6372) < 5e-3
```

## 3. 对照实验设计

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 二分类 $Q = (t, 1-t)$，$t$ 扫过 $[0.001, 0.999]$ | $P = (0.7, 0.3)$ 固定 | $H(P, Q)$、$D_{\mathrm{KL}}(P\|Q)$、$H(P)$ 三条曲线 |
| E2 | $(P, Q)$ 交换方向 | 同一批分布 | $D_{\mathrm{KL}}(P\|Q)$ vs $D_{\mathrm{KL}}(Q\|P)$ 散点图（偏离对角线程度） |
| E3 | 真实概率 $p = 0.7$ 的二项样本，预测 $q$ 扫过 $[0.01, 0.99]$ | 每个格点用 $10^5$ 个 MC 样本 | CE 与 Brier 的期望损失曲线、最小值位置 |
| E4 | 模型对真类概率 $q_y \in \{1, 0.9, 0.5, 0.1, 0.01, 10^{-6}\}$ | — | CE 与 Brier 损失数值、比值 |
| E5 | 连续高斯对：$(\mu_1,\sigma_1)$ 固定 $(0,1)$，$(\mu_2, \sigma_2)$ 网格 | 格点离散化 $\Delta = 0.01$ | 离散 KL 热力图 vs 1.5 闭式解 |

## 4. 预期图表

- 图 1（E1）：$t$ 横轴，三条曲线：$H(P,Q)$ 与 $D_{\mathrm{KL}}(P\|Q)$ 都在 $t = 0.7$ 处取最小；$H(P,Q)$ 的最小值恰好落在水平线 $H(P) \approx 0.611$ 上（恒等式的可视化：CE 曲线 = KL 曲线整体抬升 $H(P)$）。
- 图 2（E2）：散点图 20 个随机分布对，$D_{\mathrm{KL}}(P\|Q)$ vs $D_{\mathrm{KL}}(Q\|P)$，对角线参考。所有点离对角线明显偏离，两轴量级不同。
- 图 3（E3）：两条期望损失曲线，均在 $q = 0.7$ 处取最小；CE 曲线在两端（$q \to 0$）急剧翘起，Brier 曲线是温和的抛物线。
- 图 4（E4）：柱状图对比，最后一组 $q_y = 10^{-6}$：CE $\approx 13.8$，Brier $\approx 1$。
- 图 5（E5）：二维热力图，闭式解的等高线叠加在离散数值上，应重合。

## 5. 观测指标（metrics.json）

```json
{
  "E1_min_positions": {"H_PQ_argmin_t": 0.7, "KL_argmin_t": 0.7, "H_P_at_min": ...},
  "E2_asymmetry": {"max_abs_diff": ..., "mean_forward": ..., "mean_backward": ...},
  "E3_proper_check": {"CE_argmin_q": ..., "Brier_argmin_q": ...},
  "E4_table": {"qy_1": {"CE": 0.0, "Brier": 0.0}, "qy_1e-6": {"CE": ..., "Brier": ...}},
  "E5_gaussian_kl_max_rel_err": ...,
  "T_all_passed": true
}
```

## 6. 结果分析

1. E1 中 $H(P, Q)$ 的最小值为什么恰好等于 $H(P)$？用恒等式解释，并说明这句话等价于「交叉熵不可能低于熵」。
2. E2 中找出一对分布使得前向远大于反向，和另一对相反。分别对应 $Q$ 的什么形态（少给/多给质量）？
3. E3 两条曲线都最小在真实概率处（proper），那 CE 的优势到底在哪？结合 E4 的数值表回答：训练早期模型常输出接近 one-hot 的错误预测，两种损失此时的梯度表现差多少倍？
4. 期望 CE 曲线（E3）与图 1 的 $H(P,Q)$ 曲线是什么关系？（应认出它们是同一条曲线的两种视角：期望 NLL = 交叉熵。）
5. E5 中离散近似误差主要来自哪里？格点 $\Delta$ 从 0.01 加密到 0.001，误差如何变化？

## 7. 验收清单

- [ ] T1-T6 全部通过
- [ ] 能白板推导 $H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q)$ 和 KL 非负（Jensen 路线）
- [ ] 能口头解释前向/反向 KL 的 mode-covering / mode-seeking 差异，并各举一个后续课程的用例（VAE / RLHF）
- [ ] 能解释「训练集平均 NLL = 模型与经验分布的交叉熵」这一句话的每个成分
- [ ] E1、E3 的最小值位置与理论值吻合（误差小于一个格点步长）

## 8. 概念接口预告

- 实验 6（MNIST 分类）：loss 曲线 = 本实验的 CE，逐 batch 的 NLL 均值可直接对照。
- 实验 15（VAE）：KL 项 = $D_{\mathrm{KL}}(q(z|x) \| p(z))$，两个高斯的闭式解即 1.5。
- 实验 30（PPO）：KL 惩罚 = $D_{\mathrm{KL}}(\pi_{\mathrm{new}} \| \pi_{\mathrm{old}})$。
- 实验 39（DPO）：隐式奖励由 policy 与 reference 的对数比构成，其期望即反向 KL 的控制项。
