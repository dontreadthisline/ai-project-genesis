---
title: "Lab 02：联合分布与条件分布"
lab: 2
---

对应大纲实验 2。依赖：Lab 01 的 `distributions.py`。产出：`run_lab02.py`。

## 0. 理论问题

1. 联合、边缘、条件三种分布之间如何换算？为什么边缘分布相同不代表联合分布相同？
2. 条件分布 $p(y \mid x)$ 如何随相关系数 $\rho$ 变化？为什么条件方差一定不超过边缘方差？
3. 如何从联合样本中「经验地」估计条件分布？这种估计的代价是什么？

## 1. 数学基础

### 1.1 三种分布的换算

联合密度 $p(x, y)$、边缘密度、条件密度：

$$
p(x) = \int p(x, y)\, dy, \qquad p(y \mid x) = \frac{p(x, y)}{p(x)}, \qquad p(x, y) = p(x)\, p(y \mid x)
$$

独立性 $p(x,y) = p(x)p(y)$ 等价于条件分布不随 $x$ 变化：$p(y \mid x) = p(y)$。

关键事实（本实验的核心）：边缘是对联合的积分，积分会丢失「耦合结构」。同一个边缘 $p(x)$、$p(y)$ 可以配出无穷多种联合。

### 1.2 二维高斯的条件分布（推导）

设

$$
\begin{pmatrix} x \\ y \end{pmatrix} \sim \mathcal N\!\left( \begin{pmatrix} \mu_x \\ \mu_y \end{pmatrix}, \begin{pmatrix} \sigma_x^2 & \rho\, \sigma_x \sigma_y \\ \rho\, \sigma_x \sigma_y & \sigma_y^2 \end{pmatrix} \right)
$$

用精度矩阵（协方差矩阵的逆）推导。先求逆：

$$
\Sigma^{-1} = \frac{1}{1 - \rho^2}
\begin{pmatrix} 1/\sigma_x^2 & -\rho/(\sigma_x \sigma_y) \\ -\rho/(\sigma_x \sigma_y) & 1/\sigma_y^2 \end{pmatrix}
= \begin{pmatrix} \Lambda_{xx} & \Lambda_{xy} \\ \Lambda_{xy} & \Lambda_{yy} \end{pmatrix}
$$

联合密度的指数项（去掉常数）为 $-\frac12 \Lambda_{xx}(x-\mu_x)^2 - \Lambda_{xy}(x-\mu_x)(y-\mu_y) - \frac12 \Lambda_{yy}(y-\mu_y)^2$。

固定 $x$，把它看作 $y$ 的函数并配方。设条件分布为 $\mathcal N(m, s^2)$，则指数应为 $-\frac{1}{2s^2}(y - m)^2 + \text{const}(x)$。比较 $y^2$ 项系数：

$$
\frac{1}{s^2} = \Lambda_{yy} = \frac{1}{\sigma_y^2 (1 - \rho^2)}
\quad\Longrightarrow\quad
s^2 = \sigma_y^2 (1 - \rho^2)
$$

比较 $y$ 一次项系数：展开后一次项总系数为 $\Lambda_{yy}\mu_y - \Lambda_{xy}(x - \mu_x)$，而 $\mathcal N(m, s^2)$ 的一次项系数为 $m/s^2$，故 $m/s^2 = \Lambda_{yy}\mu_y - \Lambda_{xy}(x - \mu_x)$，解出：

$$
m = \mu_y - \frac{\Lambda_{xy}}{\Lambda_{yy}} (x - \mu_x) = \mu_y + \rho\, \frac{\sigma_y}{\sigma_x} (x - \mu_x)
$$

结论（Lab 19 扩散模型条件去噪、Lab 25 Q-Learning 的条件期望结构都与此同构）：

$$
y \mid x \sim \mathcal N\!\left( \mu_y + \rho \frac{\sigma_y}{\sigma_x}(x - \mu_x),\; \sigma_y^2 (1 - \rho^2) \right)
$$

三个直接推论：

1. 条件均值是 $x$ 的线性函数，斜率 $\rho\, \sigma_y / \sigma_x$（线性回归的总体版本）。
2. 条件方差 $\sigma_y^2(1-\rho^2) \le \sigma_y^2$：知道 $x$ 之后 $y$ 的不确定性只会减少或不增。$|\rho| \to 1$ 时条件方差趋于 0，即由 $x$ 可几乎确定 $y$。
3. 方差缩减量 $\rho^2 \sigma_y^2$ 正是「被 $x$ 解释掉的方差」，$\rho^2$ 即决定系数。

<iframe src="resources/figures/conditional-slice.html" style="width:100%;height:404px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="条件分布 = 联合的竖切"></iframe>

<p align="center"><b>图 02.1</b>：联合云的竖切给出条件分布（ρ=0.8）</p>

### 1.3 相同边缘、不同联合：两个反例

**反例 A（高斯族内部，靠 $\rho$ 区分）**：$\rho$ 取不同值时边缘都还是 $\mathcal N(0,1)$（二维高斯边缘是一维高斯，与 $\rho$ 无关），联合形状完全不同。说明：边缘相同 + 每个分量是高斯，联不起来。

**反例 B（高斯混合，彻底走出高斯族）**：

$$
p(x, y) = 0.5\, \mathcal N\!\left(\mathbf 0, \Sigma_+\right) + 0.5\, \mathcal N\!\left(\mathbf 0, \Sigma_-\right), \qquad
\Sigma_\pm = \begin{pmatrix} 1 & \pm 0.9 \\ \pm 0.9 & 1 \end{pmatrix}
$$

每个分量的边缘是 $\mathcal N(0,1)$，混合后边缘 $= 0.5\,\mathcal N(0,1) + 0.5\,\mathcal N(0,1) = \mathcal N(0,1)$，与反例 A 的边缘完全相同。但联合分布呈 X 形（两条脊线），协方差 $= 0.5\Sigma_+ + 0.5\Sigma_- = I$：整体线性相关为零，结构全在非线性依赖里。这就是「皮尔逊相关为零但强依赖」的标准例子。

<iframe src="resources/figures/same-marginals.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="相同边缘、不同联合"></iframe>

<p align="center"><b>图 02.2</b>：X 形混合: 边缘相同、联合相反</p>

### 1.4 条件分布的经验估计

给定联合样本 $(x_i, y_i)$，估计 $p(y \mid x = x_0)$：取落在窄带 $x \in [x_0 - \delta, x_0 + \delta]$ 内的样本，做其 $y$ 值的直方图。

代价分析：窄带内期望样本数约为 $n \cdot p_x(x_0) \cdot 2\delta$。$\delta$ 太大引入偏差（把 $x \ne x_0$ 的条件混进来，称为 smoothing bias）；$\delta$ 太小则带内样本不足、方差爆炸。这与 Lab 01 的 bin 宽两难本质相同：任何「用局部样本逼近条件」的估计器都要在偏差与方差之间取舍。

## 2. 实现任务

在 `distributions.py` 中补充：

实现清单（模块与接口见源码）：

- `gaussian_conditional` — 已知多维高斯 N(mu, Sigma)，观测其中 obs_idx 维为 x_obs，
- `gaussian_mixture_sample` — 从混合高斯采样：先按 weights 抽成分，再从对应高斯采样。
- `empirical_conditional` — 经验条件分布（讲义 §1.4）：取窄带 |samples[:,obs_idx] - x0| <= delta 内的样本。

- 模块源码：[`code/lab02-joint-and-conditional/`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab02-joint-and-conditional/)
- 自测断言（红灯标准）：`run_lab02.py` 的 `check_implementation()`
- 实验入口：`cd code/lab02-joint-and-conditional && uv run run_lab02.py`


## 3. 概念接口预告

$p(y \mid x)$ 的观点在后续课程中连续变形：Lab 19 中 $x_t \mid x_0$ 与 $x_{t-1} \mid x_t$ 是高斯条件链；Lab 25 中 $Q(s' \mid s, a)$ 是离散条件分布；Lab 37 中 SFT 学的是 $p(\text{response} \mid \text{prompt})$。本实验建立的「条件 = 联合除以边缘、条件分布比边缘窄」的直觉会一直复用。
