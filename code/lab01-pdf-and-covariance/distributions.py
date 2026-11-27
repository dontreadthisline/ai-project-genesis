"""Lab 01 分布模块：一维/二维高斯的密度与采样（MLX 实现）。

实现要求与公式见 docs/stage1-probability/lab01-pdf-and-covariance.md（§1.2、§1.3、§2）。
禁止调用 scipy；数值实现用 mx.linalg.slogdet 与 mx.linalg.solve，禁止 mx.linalg.inv。
随机数：脚本开头 mx.random.seed(0)，函数内直接 mx.random.normal 等。
"""

import mlx.core as mx


def univariate_gaussian_pdf(x, mu: float, sigma: float) -> mx.array:
    """一维高斯密度（讲义 §1.2）：

        N(x | mu, sigma^2) = 1/sqrt(2*pi*sigma^2) * exp(-(x-mu)^2 / (2*sigma^2))

    x 为 mx.array（或标量），返回同形状密度值。
    """
    p_x = (
        1.0
        / (mx.sqrt(mx.array(2.0 * mx.pi)) * sigma)
        * mx.exp(-((x - mu) ** 2) / (2 * sigma**2))
    )
    return mx.array(p_x)


def multivariate_gaussian_pdf(xs, mu, Sigma: mx.array) -> mx.array:
    """多维高斯密度（讲义 §1.3）。xs: (N, d)，mu: (d,)，Sigma: (d, d)，返回 (N,)。

    实现提示（返回的是密度值；内部路线自选——二维直接 det 完全可行，
    log 域是高维/训练场景的推荐习惯，见 lab04/lab15）:
        sign, logdet = mx.linalg.slogdet(Sigma, stream=mx.cpu)  # 返回 (符号, log|det|)
        det = sign * mx.exp(logdet)                             # 或直接 mx.linalg.det（二维安全）
        quad = 对每个样本解 Sigma^{-1}(x-mu)：mx.linalg.solve(Sigma, (xs-mu).T, stream=mx.cpu)
        pdf = exp(-0.5*quad) / ((2*pi)^{d/2} * sqrt(det))
    """
    d = Sigma.shape[0]
    mean = xs - mu
    inv = mx.linalg.solve(Sigma, mx.eye(d, d), stream=mx.cpu)
    _, det = mx.linalg.slogdet(Sigma, stream=mx.cpu)
    coeff = 1 / (mx.power(2 * mx.pi, d / 2) * mx.sqrt(det))
    return coeff * mx.exp(-1 / 2.0 * mean.T @ inv @ mean)


def sample_multivariate_gaussian(mu, Sigma, n: int) -> mx.array:
    """从 N(mu, Sigma) 采样 n 个样本（线性变换法，推导见讲义 §1.5）：

        z ~ N(0, I)（mx.random.normal((n, d))），x = mu + L z，
        其中 Sigma = L L^T（mx.linalg.cholesky(Sigma, stream=mx.cpu)）

    返回 (n, d)。为什么这样得到的协方差正好是 Sigma？答案写入实验报告（讲义 §6 问题 4）。
    """
    raise NotImplementedError("TODO(lab01 §2): Cholesky 线性变换")
