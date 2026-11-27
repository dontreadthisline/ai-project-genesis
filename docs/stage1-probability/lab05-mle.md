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

- 模块源码：[`code/lab05-mle/`](../../code/lab05-mle/)
- 自测断言（红灯标准）：`run_lab05.py` 的 `check_implementation()`
- 实验入口：`cd code/lab05-mle && uv run run_lab05.py`


## 3. 对照实验设计

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $n \in \{5, 10, 30, 100, 1000, 10^4\}$ | 真值 $(\mu, \sigma^2) = (3, 4)$，每档重复 2000 次 | $\hat\mu, \hat\sigma^2$ 的抽样分布直方图、偏差、方差、MSE |
| E2 | 估计量：$\hat\sigma^2_{1/n}$ vs $\hat\sigma^2_{1/(n-1)}$ | 同 E1 | 两者的经验偏差随 $n$ 曲线 |
| E3 | $n \in \{10, 100, 1000\}$ | 同 E1 | $\hat\mu$ 经验标准差 vs 理论 $\sigma/\sqrt n$ |
| E4 | 先验强度 $\tau^2 \in \{10^2, 1, 10^{-2}\}$，$n \in \{5, 50, 500\}$ | $\mu_0 = 0$（故意设偏离真值 3） | $\hat\mu_{\mathrm{MAP}}$ 相对 MLE 的收缩量 |
| E5 | 模型错设：数据改为污染高斯 90% $\mathcal N(3, 4)$ + 10% $\mathcal N(3, 36)$（重尾，MLX 直接可实现：对 10% 的样本乘 3 倍噪声） | 拟合仍是高斯 MLE | $\hat\sigma^2$ 收敛到何处、对离群点的敏感性 |

## 4. 预期图表

- 图 1（E1）：六联直方图（$\hat\mu$ 一行、$\hat\sigma^2$ 一行）。预期：$\hat\mu$ 全部对称钟形且随 $n$ 收窄；$\hat\sigma^2$ 在 $n = 5$ 时右偏严重（方差估计量的支撑是正的），$n$ 大时趋近正态。
- 图 2（E2）：偏差 vs $n$ 曲线（对数横轴）。预期 $1/n$ 版本偏差 $\approx -\sigma^2/n$（$n=5$ 时约 $-0.8$，可从 2000 次均值直接读出），$1/(n-1)$ 版本在 0 附近抖动。
- 图 3（E3）：$\hat\mu$ 经验标准差与理论线 $\sigma/\sqrt n = 2/\sqrt n$ 重合（对数-对数下斜率 $-1/2$，与 Lab 03 图 1 呼应）。
- 图 4（E4）：$3 \times 3$ 热力图或折线族：MAP 随 $\tau^2$ 与 $n$ 的收缩行为。预期：强先验 + 小样本时 MAP 明显偏离 MLE 被拉向 0；$n = 500$ 时所有 $\tau^2$ 下 MAP 与 MLE 几乎重合。
- 图 5（E5）：污染高斯数据上 $\hat\sigma^2$ 的抽样分布 vs 干净高斯数据上的对照。混合方差 $= 0.9 \times 4 + 0.1 \times 36 = 8.4$，预期 $\hat\sigma^2$ 稳定收敛到 8.4 附近而非 4，分布宽得多——说明 MLE 无偏性只在模型正确设定时成立，错设下它收敛到 KL 投影的参数。

## 5. 观测指标

以下指标在实验运行后写入 `output/metrics.json`（键名与 §3 对照实验表的因变量对应；写入逻辑见 `run_lab05.py`）：




## 6. 结果分析

1. E2 中把经验偏差与理论 $-\sigma^2/n$ 曲线叠加，是否吻合？$n = 5$ 时读出的偏差是多少？
2. E3 的经验标准差与 $\sigma/\sqrt n$ 的比值应接近 1。把它与 Lab 03 的标准误差公式放在一起，说明「估计量的标准差」和「Monte Carlo 标准误差」为什么是同一个数学对象。
3. E4 中要让 MAP 偏离 MLE 10% 以上，需要多强的先验或多小的 $n$？由此评价「先验在数据足够多时几乎不起作用」这句话的量化含义。
4. E5 中模型错设时 MLE 还在「最小化什么」？（用 1.5 回答：仍是与经验分布 KL 最小的高斯，即投影；它只是不再收敛到真实参数。）这一观点在实验 35（语言模型）中复用：模型永远只是在做 KL 投影。
5. 用一句话向同伴解释 1.5 的三级等价（NLL 均值 = 交叉熵 = KL），并指出这个等价在哪一步用到了「i.i.d.」。

## 7. 验收清单

- [ ] T1-T3 全部通过
- [ ] 能白板完成 1.2 的两个求导步骤和 1.3 的偏差推导
- [ ] 能解释 $1/n$ 与 $1/(n-1)$ 之争的来源，并知道何时该用哪个
- [ ] 能写出并解释 $\hat\mu \sim \mathcal N(\mu, \sigma^2/n)$，说明它与 Lab 03 的 $1/\sqrt n$ 收敛的统一性
- [ ] 能复述 1.5 的推导链条（三级等价），并说出它在第二阶段 MNIST 训练中的具体形态
- [ ] E2、E3 的经验曲线与理论曲线吻合（比值在 $0.95 \sim 1.05$）

## 8. 阶段收束

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
