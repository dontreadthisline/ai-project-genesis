---
title: "Lab 03：Monte Carlo 估计"
lab: 3
---

对应大纲实验 3。依赖：NumPy。产出：`experiments/lab03_monte_carlo.py`。本实验是后续策略梯度（实验 28/30/45）、ELBO 采样估计（实验 15）、reward 期望估计（实验 44）的直接数学前置。

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

```python
import numpy as np

def mc_estimate(f, sampler, n: int, rng):
    """f: 向量化函数 np.ndarray -> np.ndarray；sampler: callable(n, rng) -> np.ndarray。
    返回 dict(est=float, se=float, n=int)。
    要求 se 用 1/(n-1) 的样本方差公式。"""
    xs = sampler(n, rng)
    fs = f(xs)
    # TODO
    ...
```

自测（解析解已知的对照用例）：

```python
rng = np.random.default_rng(0)
std_normal = lambda n, rng: rng.standard_normal(n)

# T1: E[X] = 0, E[X^2] = 1（n = 10^6，2 个标准误差内）
r1 = mc_estimate(lambda z: z, std_normal, 10**6, rng)
assert abs(r1["est"] - 0.0) < 2 * r1["se"]
r2 = mc_estimate(lambda z: z**2, std_normal, 10**6, rng)
assert abs(r2["est"] - 1.0) < 2 * r2["se"]

# T3: E[|X|] = sqrt(2/pi)
r3 = mc_estimate(lambda z: np.abs(z), std_normal, 10**6, rng)
assert abs(r3["est"] - np.sqrt(2/np.pi)) < 2 * r3["se"]

# T4: E[sin X] = 0（奇函数，对称分布）
r4 = mc_estimate(lambda z: np.sin(z), std_normal, 10**6, rng)
assert abs(r4["est"]) < 2 * r4["se"]

# T5: pi/4，先验误差检查
square = lambda n, rng: rng.uniform(-1, 1, size=(n, 2))
r5 = mc_estimate(lambda P: (P**2).sum(axis=1) <= 1, square, 10**6, rng)
assert abs(r5["est"] - np.pi/4) < 5e-3
```

## 3. 对照实验设计

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $n \in \{10^1, 10^2, \dots, 10^6\}$ | $f(x) = x$，$X \sim \mathcal N(0,1)$ | $\lvert\hat\mu_n - \mu\rvert$、$\hat{\mathrm{se}}_n$（重复 100 次取均值） |
| E2 | $f \in \{x+3,\ x^2,\ |x|,\ \mathbb 1[x > 3]\}$（期望均非零：$3,\ 1,\ \sqrt{2/\pi},\ 0.00135$） | $n = 10^5$，重复 1000 次 | 估计量的经验标准差、理论 $\sigma_f/\sqrt n$、相对误差 |
| E3 | 稀有事件 $f = \mathbb 1[x > c]$，$c \in \{1, 2, 3, 4\}$ | $n = 10^6$，重复 200 次 | 相对误差 vs $p_c$，与 $1/\sqrt{n p_c}$ 曲线对照 |
| E4 | $\pi$ 估计 | $n$ 从 $10^2$ 到 $10^7$ | $\lvert\hat\pi - \pi\rvert$（对数轴） |

## 4. 预期图表

- 图 1（E1）：双对数坐标，$n$ vs 平均绝对误差。预期斜率约 $-1/2$（拟合值应在 $-0.45 \sim -0.55$ 之间）；同一张图叠加理论标准误差 $\sigma_f/\sqrt n$ 曲线，两线应重合。
- 图 2（E2）：四联图，每图是 1000 次重复估计量的直方图 + 正态密度曲线 $\mathcal N(\mu, \sigma_f^2/n)$ 叠加。预期平滑函数（$x+3$、$x^2$、$|x|$）的直方图与正态曲线贴合；$\mathbb 1[x>3]$ 的直方图明显右偏（泊松型），且相对误差比其他函数大一到两个数量级。
- 图 3（E3）：对数-对数图，$p_c$ vs 相对误差，叠加 $1/\sqrt{n p_c}$ 理论线。预期落在理论线附近：$c = 4$ 时 $p \approx 3.2\times 10^{-5}$，$n = 10^6$ 下期望命中仅约 32 次，相对误差约 18%；$c = 5$ 时期望命中约 0.29 次，多数重复实验中一个命中都没有，估计量为 0。
- 图 4（E4）：$\pi$ 估计误差随 $n$ 收敛曲线 + $1/\sqrt n$ 参考线。

## 5. 观测指标（metrics.json）

```json
{
  "E1_loglog_slope": ...,
  "E2_per_function": {
    "x":    {"emp_sd": ..., "theo_se": ..., "rel_err": ...},
    "x2":   {...}, "ind_x_gt_3": {...}, "sin": {...}
  },
  "E3_rare_event": {"c=1": {"p_hat": ..., "rel_err": ...}, "c=2": {...}, "c=3": {...}, "c=4": {...}},
  "T_all_passed": true
}
```

## 6. 结果分析

1. E1 拟合出的斜率是多少？与 $-1/2$ 的偏差来自哪里（提示：绝对值的期望 $\mathbb E|\hat\mu - \mu|$ 与标准误差只差一个 $\sqrt{2/\pi}$ 因子）？
2. E2 中 $\mathbb 1[x > 3]$ 的相对误差比 $|x|$ 大约多少倍？用 1.4 的公式先算再验证。这解释了 RL 中为什么稀疏奖励（实验 44）难训练。
3. E2 直方图验证 CLT 时，$\mathbb 1[x>3]$ 的直方图为什么不对称？CLT 要求多大的 $n$ 才让偏态消失（可加做 $n = 10^6$ 的重复实验观察）？
4. $\hat{\mathrm{se}}$ 与重复实验的经验标准差的比值应接近 1。如果实现里误用 $1/n$ 而非 $1/(n-1)$，在 $n = 10$ 时这个比值会偏多少？（顺带解释为什么无偏方差估计用 $n-1$。）

## 7. 验收清单

- [ ] T1-T5 全部通过
- [ ] 能白板推导 $\mathrm{Var}[\hat\mu_n] = \sigma_f^2/n$ 与标准误差公式
- [ ] 能解释 $1/\sqrt n$ 收敛的业务含义：样本量翻倍，精度只提升 $\sqrt 2$ 倍
- [ ] E1 的 log-log 斜率落在 $[-0.55, -0.45]$
- [ ] 能定量解释稀有事件的相对误差公式，并算出「10% 相对误差需要多少样本」
- [ ] CLT 验证图（图 2）中至少三个函数的经验分布与正态曲线明显吻合

## 8. 进一步实验（选做）

被积函数与采样分布不匹配的对照：估计 $\mathbb E_{X \sim \mathcal N(0,1)}[e^{X/2}]$（解析解 $e^{1/8}$），再故意改用 $\mathcal N(0, 3)$ 采样但不加权重（错误做法）与加上重要性权重 $w = p(x)/q(x)$ 的正确做法对比。观察权重的方差如何影响估计稳定性。此实验为选做，重要性采样正式进入主线在 RL 阶段（PPO 的 ratio）。
