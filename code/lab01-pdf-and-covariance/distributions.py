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
    raise NotImplementedError("TODO(lab01 §1.2): 按公式实现")


def multivariate_gaussian_pdf(xs, mu, Sigma) -> mx.array:
    """多维高斯密度（讲义 §1.3）。xs: (N, d)，mu: (d,)，Sigma: (d, d)，返回 (N,)。

    实现提示：
        _, logdet = mx.linalg.slogdet(Sigma, stream=mx.cpu)  # linalg 不支持 GPU，显式 CPU
        quad = 对每个样本解 Sigma^{-1}(x-mu)：mx.linalg.solve(Sigma, (xs-mu).T, stream=mx.cpu)
        log_pdf = -0.5*(d*log(2*pi) + logdet + 二次型)
    """
    raise NotImplementedError("TODO(lab01 §1.3): 用 slogdet + solve 实现")


def sample_multivariate_gaussian(mu, Sigma, n: int) -> mx.array:
    """从 N(mu, Sigma) 采样 n 个样本（Cholesky 方法）：

        z ~ N(0, I)（mx.random.normal((n, d))），x = mu + L z，
        其中 Sigma = L L^T（mx.linalg.cholesky(Sigma, stream=mx.cpu)）

    返回 (n, d)。为什么这样得到的协方差正好是 Sigma？答案写入实验报告（讲义 §6 问题 4）。
    """
    raise NotImplementedError("TODO(lab01 §2): Cholesky 线性变换")
