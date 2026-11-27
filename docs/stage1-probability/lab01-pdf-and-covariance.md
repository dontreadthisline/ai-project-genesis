---
title: "Lab 01：一维与二维概率分布"
lab: 1
---

对应大纲实验 1。依赖：MLX（实现）、NumPy（仅画图桥接）、Matplotlib。产出：`run_lab01.py`、图表、`metrics.json`。

## 0. 理论问题

1. 概率密度函数（PDF）与采样直方图是什么关系？为什么密度值可以大于 1？
2. 均值和方差在高斯密度曲线上对应什么几何特征？
3. 协方差矩阵如何描述变量之间的相关性？等密度轮廓为什么是椭圆，椭圆的朝向和长短轴由什么决定？

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

推论（实验中要亲眼验证）：密度可以取大于 1 的值。例如 $\sigma = 0.01$ 的高斯在峰值处 $p(\mu) = 1/(\sqrt{2\pi}\,\sigma) \approx 39.9$，但概率质量仍然只有 1。

### 1.2 一维高斯与归一化常数

$$
\mathcal N(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
$$

归一化常数来自高斯积分。设 $I = \int_{-\infty}^{+\infty} e^{-x^2/2}\, dx$，则

$$
I^2 = \int\!\!\int e^{-(x^2+y^2)/2}\, dx\, dy = \int_0^{2\pi}\!\!\int_0^{+\infty} e^{-r^2/2} r\, dr\, d\theta = 2\pi \cdot \left[ -e^{-r^2/2} \right]_0^{+\infty} = 2\pi
$$

故 $I = \sqrt{2\pi}$。这个常数后续会反复出现：多维高斯的归一化常数 $(2\pi)^{-d/2}|\Sigma|^{-1/2}$、KL 散度的解析公式、ELBO 的重构项，全部由它派生。

均值 $\mu$ 是密度的对称中心和峰值位置；方差 $\sigma^2$ 控制宽度。峰值高度 $1/(\sqrt{2\pi}\,\sigma)$ 与 $\sigma$ 成反比：越宽越矮，总面积不变。

### 1.3 二维高斯与协方差矩阵的几何

$$
\mathcal N(\mathbf x \mid \boldsymbol\mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\!\left( -\tfrac12 (\mathbf x - \boldsymbol\mu)^T \Sigma^{-1} (\mathbf x - \boldsymbol\mu) \right)
$$

协方差矩阵 $\Sigma$ 对称半正定（正定时可逆）。对其做特征分解：

$$
\Sigma = U \Lambda U^T, \qquad \Lambda = \mathrm{diag}(\lambda_1, \lambda_2), \quad U = [u_1, u_2], \quad U U^T = I
$$

代入指数的二次型：

$$
(\mathbf x - \boldsymbol\mu)^T \Sigma^{-1} (\mathbf x - \boldsymbol\mu)
= \sum_{i=1}^{2} \frac{\big( u_i^T (\mathbf x - \boldsymbol\mu) \big)^2}{\lambda_i}
$$

取指数为常数 $c^2$ 得等密度集合：

$$
\sum_i \frac{\big( u_i^T (\mathbf x - \boldsymbol\mu) \big)^2}{\lambda_i} = c^2
$$

这是一个中心在 $\boldsymbol\mu$ 的椭圆：主轴方向为特征向量 $u_i$，半轴长度为 $c\sqrt{\lambda_i}$。

标准相关系数形式的二维协方差：

$$
\Sigma = \begin{pmatrix} \sigma_x^2 & \rho\, \sigma_x \sigma_y \\ \rho\, \sigma_x \sigma_y & \sigma_y^2 \end{pmatrix}
$$

$\rho \in [-1, 1]$。$\rho = 0$ 时椭圆轴与坐标轴对齐（独立）；$|\rho| \to 1$ 时椭圆退化为一条直线方向 $u \propto (\sigma_x, \mathrm{sign}(\rho)\, \sigma_y)$。椭圆长轴倾角：

$$
\theta = \tfrac12 \arctan\!\left( \frac{2\rho\, \sigma_x \sigma_y}{\sigma_x^2 - \sigma_y^2} \right)
$$

### 1.4 直方图密度估计

对 $n$ 个样本、区间宽度 $h$ 的一个 bin，落入的期望频数为 $n \cdot p(x_{\text{bin}}) \cdot h$。因此

$$
\hat p(x_{\text{bin}}) = \frac{\text{频数}}{n\, h}
$$

即 `plt.hist(..., density=True)` 的含义。bin 宽 $h$ 的两难：$h$ 太大则估计偏倚大（抹平结构），$h$ 太小则每个 bin 样本少、随机波动大（方差大）。这是偏差-方差权衡的最直观版本，后续在核密度、生成模型评估中都会重现。

### 1.5 从分布采样：线性变换法

到此为止分布是「被观察的数学对象」（密度、几何）；而 E3/E5 需要反方向的操作：**从分布生成样本**。

手上只有标准正态伪随机数（`mx.random.normal`），如何得到 $\mathcal N(\mu, \Sigma)$ 的样本？线性变换。设 $z \sim \mathcal N(0, I_d)$，取 $x = \mu + A z$，则

$$
\mathbb E[x] = \mu + A\,\mathbb E[z] = \mu, \qquad
\mathrm{Cov}[x] = \mathbb E\big[(x-\mu)(x-\mu)^T\big] = A\, \mathbb E[zz^T]\, A^T = A A^T
$$

取 $A = L$，其中 $\Sigma = L L^T$（Cholesky 分解，$L$ 下三角，任何对称正定矩阵都存在），即得 $\mathrm{Cov}[x] = \Sigma$；而高斯分布由前两阶矩完全决定，故 $x \sim \mathcal N(\mu, \Sigma)$。

几何视角：这与 §1.3 的特征分解同族——$U\sqrt{\Lambda}$ 与 $L$ 都在把各向同性的标准正态圆云「拉伸、旋转」成目标椭球（$L$ 是不旋转的斜切版）。采样 = 在圆云里取一点，再随云一起变形。

这是「伪随机数 + 确定性变换 = 目标分布样本」的最小例子。Lab 03 的 Monte Carlo 全部建立在此之上；Lab 15/16 的重参数化是同一思想在梯度传播中的翻版；一般分布的采样方法（逆变换、拒绝采样）超出本讲需要，不展开。

## 2. 实现任务

创建 `distributions.py`：

实现清单（模块与接口见源码）：

- `univariate_gaussian_pdf` — 一维高斯密度。x: mx.array 或标量。禁止调用 scipy。
- `multivariate_gaussian_pdf` — 多维高斯密度。xs: (N, d)，mu: (d,)，Sigma: (d, d)，返回 (N,)。
- `sample_multivariate_gaussian` — 从 N(mu, Sigma) 采样 n 个点：先采样 z ~ N(0, I)（mx.random.normal），

- 模块源码：[`code/lab01-pdf-and-covariance/`](../../code/lab01-pdf-and-covariance/)
- 自测断言（红灯标准）：`run_lab01.py` 的 `check_implementation()`
- 实验入口：`cd code/lab01-pdf-and-covariance && uv run run_lab01.py`


## 3. 对照实验设计

| 编号 | 自变量 | 控制变量 | 因变量（记录项） |
|---|---|---|---|
| E1 | $\mu \in \{-2, 0, 3\}$ | $\sigma = 1$ | 密度曲线位置 |
| E2 | $\sigma \in \{0.2, 1, 5\}$ | $\mu = 0$ | 峰高、宽度、密度峰值数值 |
| E3 | 样本量 $n \in \{100, 10^3, 10^4, 10^5\}$ | 同一分布 | 直方图与 PDF 曲线的最大偏差 |
| E4 | bin 数 $\in \{5, 20, 50, 200\}$，$n = 10^4$ | 同一样本 | 直方图形态（过平滑 / 抖动） |
| E5 | $\rho \in \{-0.9, -0.5, 0, 0.5, 0.9\}$，$\sigma_x = \sigma_y = 1$ | 均值 0 | 等密度椭圆朝向、样本云形态 |
| E6 | $\Sigma = \mathrm{diag}(9, 1)$ 与 $\mathrm{diag}(1, 9)$ 与旋转 45 度版本 | $\rho$ 换算 | 椭圆长短轴比、倾角 |

## 4. 预期图表

- 图 1（E1-E2）：一维 PDF 曲线族，$\mu$ 平移、$\sigma$ 变宽时峰高按 $1/\sigma$ 下降，三条曲线下面积相同。
- 图 2（E3）：四个子图，直方图（density=True）+ 真实 PDF 叠加。预期 $n = 100$ 时轮廓明显偏离，$n = 10^5$ 时贴合。
- 图 3（E4）：同一 $n$、不同 bin 宽的四联图，肉眼可见平滑偏差与随机抖动的此消彼长。
- 图 4（E5）：五行两列，左列二维直方图/散点云，右列等密度椭圆（用特征分解画 $c \in \{1, 2\}$ 的两层）。预期 $\rho$ 增大时椭圆逆时针变扁，$\rho = 0.9$ 时样本云近似沿 $y = x$ 直线分布。
- 图 5（E6）：$\mathrm{diag}(9,1)$ 的椭圆长短轴比约 $3:1$，轴沿坐标轴；旋转版倾角 45 度，轴比不变。

## 5. 观测指标

以下指标在实验运行后写入 `output/metrics.json`（键名与 §3 对照实验表的因变量对应；写入逻辑见 `run_lab01.py`）：

- `E3_max_hist_pdf_deviation.n100`
- `E3_max_hist_pdf_deviation.n1e3`
- `E3_max_hist_pdf_deviation.n1e4`
- `E3_max_hist_pdf_deviation.n1e5`
- `E5_ellipse_axis_ratio.rho_-0.9`
- `E5_ellipse_axis_ratio.rho_0`
- `E5_ellipse_axis_ratio.rho_0.9`
- `E5_sample_cov.rho_0.9`
- `T_all_passed`


## 6. 结果分析

回答以下问题（写在实验报告里）：

1. E2 中 $\sigma = 0.2$ 的峰值约为 $2.0$，密度远大于 1，这违反「概率不超过 1」吗？用 1.1 的公式解释。
2. E3 中直方图偏差随 $n$ 如何缩放？能否粗略估计出幂律指数（提示：与 Lab 03 的 $1/\sqrt n$ 对照）。
3. E5 中若只看边缘直方图（把二维样本分别投影到 x 轴和 y 轴），五种 $\rho$ 有区别吗？这说明边缘分布丢失了什么信息？（此问题直接引出 Lab 02。）
4. 用 §1.5 的推导解释 Cholesky 采样 $x = \mu + Lz$ 为什么协方差恰为 $\Sigma$；并用 T3 的采样数据实证（样本协方差随 $n$ 收敛到 $\Sigma$ 的速率）。

## 7. 验收清单

- [ ] T1-T4 全部通过，实现未调用 `scipy.stats`
- [ ] 能白板写出二维高斯密度公式，并解释每个成分（行列式、二次型）的几何来源
- [ ] 给定任意 $2\times 2$ 正定 $\Sigma$，能先手算预测椭圆长轴方向和轴比，再看图验证
- [ ] 能解释 `density=True` 的直方图纵轴含义，以及 bin 宽的偏差-方差两难
- [ ] metrics.json 完整落盘

## 8. 进一步实验（选做）

用 `sample_multivariate_gaussian` 验证仿射变换下的高斯性质：若 $X \sim \mathcal N(\mu, \Sigma)$，则 $AX + b \sim \mathcal N(A\mu + b, A\Sigma A^T)$。取 $A$ 为旋转矩阵，验证协方差矩阵随之旋转，密度形状不变。这一性质是 Lab 18（Normalizing Flow）和 Lab 20（Flow Matching）的数学前置。
