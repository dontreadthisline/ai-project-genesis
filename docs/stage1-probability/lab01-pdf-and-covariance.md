---
title: "Lab 01：一维与二维概率分布"
lab: 1
---

对应大纲实验 1。依赖：MLX。产出：`run_lab01.py`、图表、`output/metrics.json`。

## 0. 理论问题

本讲围绕三个问题展开，正文按此地图推进，§6 回收答案：

1. **密度与样本的关系**——直方图凭什么能估计 PDF？为什么密度值可以大于 1？（→ §1.1、§1.2、§1.4）
2. **参数的几何意义**——均值和方差在高斯曲线上长什么样？（→ §1.2）
3. **协方差的几何与采样的原理**——$\Sigma$ 如何决定等密度椭圆的形状与朝向？怎么从一个分布生成样本？（→ §1.3、§1.5）

## 1. 数学基础

### 1.1 概率密度与概率的关系

连续随机变量 $X$ 的密度函数 $p(x)$ 满足非负性与归一化：

$$
p(x) \ge 0, \qquad \int_{-\infty}^{+\infty} p(x)\, dx = 1
$$

概率是密度的积分，密度本身不是概率：

$$
P(x \le X \le x + dx) \approx p(x)\, dx, \qquad
P(a \le X \le b) = \int_a^b p(x)\, dx
$$

由此有一个会被反复用到的事实：**密度可以大于 1**——$\sigma = 0.01$ 的高斯峰值 $\approx 39.9$，但概率质量仍只有 1（图 1 右侧 caption 的极端情形）。

### 1.2 一维高斯与归一化常数

$$
\mathcal N(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
$$

归一化常数来自高斯积分。设 $I = \int e^{-x^2/2}\, dx$，极坐标下：

$$
I^2 = \int\!\!\int e^{-(x^2+y^2)/2}\, dx\, dy = 2\pi \int_0^{+\infty} e^{-r^2/2} r\, dr = 2\pi \ \Rightarrow\ I = \sqrt{2\pi}
$$

这个常数是全课程出镜率最高的数：多维高斯归一化、KL 闭式解、ELBO，全部由它派生。

两个参数的几何意义（**图 1**）：$\mu$ 平移曲线、$\sigma$ 同时控制宽度与峰高——越宽越矮，总面积不变。

<iframe src="resources/figures/lab01-gaussian-1d.html" style="width:100%;height:430px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="图 1：一维高斯参数的几何意义"></iframe>

<p align="center"><b>图 1</b>：一维高斯族（手绘渲染）。σ 收窄时峰高按 1/σ 抬升。</p>

### 1.3 二维高斯与协方差矩阵的几何

从一维到二维，参数从两个数升级为一个矩阵：

$$
\mathcal N(\mathbf x \mid \boldsymbol\mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\!\left( -\tfrac12 (\mathbf x - \boldsymbol\mu)^T \Sigma^{-1} (\mathbf x - \boldsymbol\mu) \right)
$$

$\Sigma$ 对称正定。对它做特征分解 $\Sigma = U \Lambda U^T$，二次型化为

$$
(\mathbf x - \boldsymbol\mu)^T \Sigma^{-1} (\mathbf x - \boldsymbol\mu) = \sum_{i} \frac{\big( u_i^T (\mathbf x - \boldsymbol\mu) \big)^2}{\lambda_i}
$$

取二次型为常数 $c^2$，等密度集合是一个中心在 $\boldsymbol\mu$ 的**椭圆**：主轴沿特征向量 $u_i$、半轴长 $c\sqrt{\lambda_i}$（**图 2**）。这是本讲最重要的几何事实——$\Sigma$ 的全部信息（方向、离散度、相关性）都画在这个椭圆上。

标准相关系数形式 $\Sigma = \begin{pmatrix} \sigma_x^2 & \rho\, \sigma_x \sigma_y \\ \rho\, \sigma_x \sigma_y & \sigma_y^2 \end{pmatrix}$，$\rho \in [-1, 1]$：$\rho = 0$ 椭圆轴与坐标轴对齐（独立）；$|\rho| \to 1$ 椭圆退化为一条直线。长轴倾角 $\theta = \tfrac12 \arctan\!\big( \frac{2\rho\, \sigma_x \sigma_y}{\sigma_x^2 - \sigma_y^2} \big)$。

<iframe src="resources/figures/lab01-covariance-ellipse.html" style="width:100%;height:460px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="图 2：协方差矩阵的几何"></iframe>

<p align="center"><b>图 2</b>：等密度椭圆。琥珀色箭头为特征向量 u₁，椭圆朝向与轴比由 Σ 唯一决定。</p>

### 1.4 直方图密度估计

有了分布（数学对象）与样本（§1.5 生成）就能提出第一个**统计**问题：用样本估计密度。对 $n$ 个样本、bin 宽 $h$，落入某 bin 的期望频数为 $n \cdot p(x)\, h$，因此

$$
\hat p(x_{\text{bin}}) = \frac{\text{频数}}{n\, h}
$$

即 `plt.hist(..., density=True)` 的含义。bin 宽的两难（**图 4**）：$h$ 大则抹平结构（偏差），$h$ 小则每格样本稀少、波动剧烈（方差）——偏差-方差权衡的最直观版本。

<iframe src="resources/figures/lab01-histogram-vs-pdf.html" style="width:100%;height:430px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="图 4：直方图密度估计"></iframe>

<p align="center"><b>图 4</b>：样本量与直方图估计。n=10⁵ 时直方图贴合真实密度曲线。</p>

### 1.5 从分布采样：线性变换法

前面分布都是「被观察的数学对象」；图 4 的样本从哪来？**手上只有标准正态伪随机数**（`mx.random.normal`），如何得到 $\mathcal N(\mu, \Sigma)$ 的样本？

线性变换：设 $z \sim \mathcal N(0, I_d)$，取 $x = \mu + A z$（期望的直观含义：平均取值；正式定义在 Lab 03），则

$$
\mathbb E[x] = \mu, \qquad \mathrm{Cov}[x] = A\,\mathbb E[zz^T]\, A^T = A A^T
$$

取 $A = L$（$\Sigma = LL^T$ 的 Cholesky 分解，$L$ 下三角，对称正定必存在）即得 $\mathrm{Cov}[x] = \Sigma$；高斯由前两阶矩完全决定，故 $x \sim \mathcal N(\mu, \Sigma)$。

几何图像（**图 3**）：与 §1.3 的特征分解同族——$U\sqrt\Lambda$ 和 $L$ 都把各向同性的圆云拉伸成目标椭球（$L$ 是不旋转的斜切版）。**采样 = 在圆云里取一点，随云一起变形。** 这一句是 Lab 03（Monte Carlo）的地基、Lab 15/16（重参数化）的前身。

<iframe src="resources/figures/lab01-sampling-transform.html" style="width:100%;height:400px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="图 3：采样即线性变换"></iframe>

<p align="center"><b>图 3</b>：标准正态圆云经 x = μ + Lz 斜切为目标椭圆云（光点沿变形方向流动）。</p>

## 2. 实现任务

结构 $784 \to 256 \to 128 \to 10$。

实现清单（签名与 TODO 见骨架源码）：

- `univariate_gaussian_pdf` — 一维高斯密度。x: mx.array 或标量。禁止调用 scipy。
- `multivariate_gaussian_pdf` — 多维高斯密度。xs: (N, d)，mu: (d,)，Sigma: (d, d)，返回 (N,)。
- `sample_multivariate_gaussian` — 从 N(mu, Sigma) 采样 n 个点（线性变换法，推导见 §1.5），

- 模块源码：[`code/lab01-pdf-and-covariance/`](../../code/lab01-pdf-and-covariance/)
- 自测断言（红灯标准）：`run_lab01.py` 的 `check_implementation()`
- 实验入口：`cd code/lab01-pdf-and-covariance && uv run run_lab01.py`

## 3. 对照实验设计

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $\mu \in \{-2, 0, 3\}$ | $\sigma = 1$ | 密度曲线位置 |
| E2 | $\sigma \in \{0.2, 1, 5\}$ | $\mu = 0$ | 峰高、宽度、密度峰值数值 |
| E3 | 样本量 $n \in \{100, 10^3, 10^4, 10^5\}$ | 同一分布 | 直方图与 PDF 曲线的最大偏差 |
| E4 | bin 数 $\in \{5, 20, 50, 200\}$，$n = 10^4$ | 同一样本 | 直方图形态（过平滑 / 抖动） |
| E5 | $\rho \in \{-0.9, -0.5, 0, 0.5, 0.9\}$，$\sigma_x = \sigma_y = 1$ | 均值 0 | 等密度椭圆朝向、样本云形态 |
| E6 | $\Sigma = \mathrm{diag}(9, 1)$ 与 $\mathrm{diag}(1, 9)$ 与旋转 45 度版本 | $\rho$ 换算 | 椭圆长短轴比、倾角 |

## 4. 预期图表

- 图 E1-E2：一维 PDF 曲线族（参照**图 1** 的形态）。
- 图 E3-E4：直方图 + PDF 叠加（参照**图 4**：小 $n$ 波动、大 $n$ 贴合）。
- 图 E5-E6：样本散点 + 等密度椭圆两层（参照**图 2**）。

## 5. 观测指标

- `E3_max_hist_pdf_deviation.n100`（及 n1e3/n1e4/n1e5）
- `E5_ellipse_axis_ratio.rho_-0.9`（及 rho_0 / rho_0.9）
- `E5_sample_cov.rho_0.9`
- `T_all_passed`

## 6. 结果分析（回扣 §0 的问题）

1.（问题 1）E2 中 $\sigma = 0.2$ 的峰值约为 $2.0$，密度远大于 1——用 §1.1 的 $p(x)\,dx$ 解释为什么这不违反概率公理。
2.（问题 1）E3 直方图偏差随 $n$ 如何缩放？与 Lab 03 将建立的 $1/\sqrt n$ 对照。
3.（问题 3）E5 只看边缘直方图（投影到单轴）时五种 $\rho$ 无差别——边缘分布丢失了什么信息？（引出 Lab 02）
4.（问题 3）用 §1.5 的推导解释 Cholesky 采样为什么协方差恰为 $\Sigma$，并用 T3 的采样数据实证收敛速率。

## 7. 验收清单

- [ ] T1-T4 通过，实现未调用 `scipy`
- [ ] 能白板写出二维高斯密度公式，并解释每个成分（行列式、二次型）的几何来源
- [ ] 给定任意 $2\times 2$ 正定 $\Sigma$，能先手算预测椭圆长轴方向和轴比，再看图验证
- [ ] 能解释 `density=True` 的直方图纵轴含义与 bin 宽的偏差-方差两难
- [ ] metrics.json 完整落盘

## 8. 进一步实验（选做）

验证仿射变换下的高斯性质：$X \sim \mathcal N(\mu, \Sigma) \Rightarrow AX + b \sim \mathcal N(A\mu + b, A\Sigma A^T)$（$A$ 取旋转矩阵）。这一性质是 Lab 18（Normalizing Flow）与 Lab 20（Flow Matching）的数学前置。
