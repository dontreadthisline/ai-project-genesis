"""Lab 05 估计器模块：高斯 MLE / NLL / MAP（MLX 实现）。

实现要求与公式见 docs/stage1-probability/lab05-mle.md（§1.2、§1.6、§2）。
"""
import mlx.core as mx


def gaussian_nll(theta: tuple[float, float], samples) -> float:
    """负对数似然均值：-mean(log N(x | mu, sigma2))。

    theta = (mu, sigma2)。内部调用 univariate_gaussian_pdf（注意它接受 sigma 而非 sigma2）。
    """
    raise NotImplementedError("TODO(lab05 §1.1)")


def gaussian_mle(samples) -> tuple[float, float]:
    """闭式解 (mu_hat, sigma2_hat)（讲义 §1.2）：

        mu_hat = mean(samples)
        sigma2_hat = mean((samples - mu_hat)^2)     # 除以 n，不是 n-1
    """
    raise NotImplementedError("TODO(lab05 §1.2)")


def gaussian_map(samples, mu0: float, tau2: float, sigma2: float) -> float:
    """已知 sigma2 时 mu 的 MAP（讲义 §1.6，高斯先验 N(mu0, tau2)）：

        mu_MAP = (n*xbar/sigma2 + mu0/tau2) / (n/sigma2 + 1/tau2)
    """
    raise NotImplementedError("TODO(lab05 §1.6)")
