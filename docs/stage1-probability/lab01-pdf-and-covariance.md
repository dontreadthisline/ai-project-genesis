---
title: "Lab 01：一维与二维概率分布"
lab: 1
---

对应大纲实验 1。依赖：NumPy、Matplotlib。产出：`experiments/lab01_1d_2d_gaussian.py`、图表、`metrics.json`。

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

## 2. 实现任务

创建 `probkit/distributions.py`：

```python
import numpy as np

def univariate_gaussian_pdf(x, mu: float, sigma: float) -> np.ndarray:
    """一维高斯密度。x: array_like。禁止调用 scipy。"""
    # TODO: 按公式 1.2 实现
    ...

def multivariate_gaussian_pdf(xs: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """多维高斯密度。xs: (N, d)，mu: (d,)，Sigma: (d, d)。
    提示：用 np.linalg.slogdet 计算行列式（数值稳定），用 np.linalg.solve 解线性方程组，
    禁止直接调用 np.linalg.inv。"""
    ...

def sample_multivariate_gaussian(rng, mu, Sigma, n: int) -> np.ndarray:
    """从 N(mu, Sigma) 采样 n 个点：先采样 z ~ N(0, I)，再线性变换 x = mu + L z，
    其中 L 是 Sigma 的 Cholesky 分解 Sigma = L L^T。说明为什么这样得到的协方差正好是 Sigma。"""
    ...
```

自测（实现正确性的判定标准，放进 `experiments/lab01_1d_2d_gaussian.py` 开头）：

```python
rng = np.random.default_rng(0)

# T1: 数值积分验证归一化（梯形法则，容差 1e-8）
x = np.linspace(-12, 12, 200_001)
assert abs(np.trapezoid(univariate_gaussian_pdf(x, 2.0, 3.0), x) - 1.0) < 1e-8

# T2: 二维密度在均值处取最大值
Sigma = np.array([[2.0, 0.6], [0.6, 1.0]])
pts = rng.normal(size=(5000, 2)) @ np.linalg.cholesky(Sigma) + np.array([1.0, -1.0])
vals = multivariate_gaussian_pdf(pts, np.array([1.0, -1.0]), Sigma)
assert multivariate_gaussian_pdf(np.array([[1.0, -1.0]]), np.array([1.0, -1.0]), Sigma)[0] >= vals.max() - 1e-12

# T3: 采样协方差收敛于 Sigma（n=200_000，逐项容差 0.05）
X = sample_multivariate_gaussian(rng, np.array([1.0, -1.0]), Sigma, 200_000)
assert np.allclose(X.mean(axis=0), [1.0, -1.0], atol=0.03)
assert np.allclose(np.cov(X.T), Sigma, atol=0.05)

# T4: 与手算对照 —— 标准正态在 0 处密度 = 1/sqrt(2*pi)
assert abs(univariate_gaussian_pdf(0.0, 0.0, 1.0) - 1/np.sqrt(2*np.pi)) < 1e-12
```

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

## 5. 观测指标（写入 metrics.json）

```json
{
  "E3_max_hist_pdf_deviation": {"n100": ..., "n1e3": ..., "n1e4": ..., "n1e5": ...},
  "E5_ellipse_axis_ratio": {"rho_-0.9": ..., "rho_0": ..., "rho_0.9": ...},
  "E5_sample_cov": {"rho_0.9": [[...], [...]]},
  "T_all_passed": true
}
```

其中直方图偏差定义为 bin 中心处 $\max_b |\hat p_b - p(x_b)|$。

## 6. 结果分析

回答以下问题（写在实验报告里）：

1. E2 中 $\sigma = 0.2$ 的峰值约为 $2.0$，密度远大于 1，这违反「概率不超过 1」吗？用 1.1 的公式解释。
2. E3 中直方图偏差随 $n$ 如何缩放？能否粗略估计出幂律指数（提示：与 Lab 03 的 $1/\sqrt n$ 对照）。
3. E5 中若只看边缘直方图（把二维样本分别投影到 x 轴和 y 轴），五种 $\rho$ 有区别吗？这说明边缘分布丢失了什么信息？（此问题直接引出 Lab 02。）
4. Cholesky 采样 $x = \mu + Lz$ 中 $z$ 各分量独立，为什么 $x$ 的协方差是 $LL^T = \Sigma$？给出推导。

## 7. 验收清单

- [ ] T1-T4 全部通过，实现未调用 `scipy.stats`
- [ ] 能白板写出二维高斯密度公式，并解释每个成分（行列式、二次型）的几何来源
- [ ] 给定任意 $2\times 2$ 正定 $\Sigma$，能先手算预测椭圆长轴方向和轴比，再看图验证
- [ ] 能解释 `density=True` 的直方图纵轴含义，以及 bin 宽的偏差-方差两难
- [ ] metrics.json 完整落盘

## 8. 进一步实验（选做）

用 `sample_multivariate_gaussian` 验证仿射变换下的高斯性质：若 $X \sim \mathcal N(\mu, \Sigma)$，则 $AX + b \sim \mathcal N(A\mu + b, A\Sigma A^T)$。取 $A$ 为旋转矩阵，验证协方差矩阵随之旋转，密度形状不变。这一性质是 Lab 18（Normalizing Flow）和 Lab 20（Flow Matching）的数学前置。
