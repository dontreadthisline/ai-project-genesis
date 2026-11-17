"""Lab 02 分布模块：在 Lab 01 基础上补充条件分布与混合高斯（MLX 实现）。

实现要求见 docs/stage1-probability/lab02-joint-and-conditional.md（§1.2、§1.3、§2）。
前三个函数在 Lab 01 已实现：把你 lab01 的完成版复制过来（保持接口一致，本实验依赖它们）。
"""
import mlx.core as mx


# ------------------------------------------------ 以下三个函数：从 lab01 复制你的实现


def univariate_gaussian_pdf(x, mu: float, sigma: float) -> mx.array:
    """Lab 01 §1.2。复制自 code/lab01-pdf-and-covariance/distributions.py。"""
    raise NotImplementedError("从 lab01 复制你的实现")


def multivariate_gaussian_pdf(xs, mu, Sigma) -> mx.array:
    """Lab 01 §1.3。复制自 lab01。"""
    raise NotImplementedError("从 lab01 复制你的实现")


def sample_multivariate_gaussian(mu, Sigma, n: int) -> mx.array:
    """Lab 01 §2。复制自 lab01。"""
    raise NotImplementedError("从 lab01 复制你的实现")


# ------------------------------------------------ 本实验新增


def gaussian_conditional(mu, Sigma, x_obs, obs_idx: int = 0):
    """已知 X ~ N(mu, Sigma)，观测第 obs_idx 维为 x_obs，返回剩余维度的 (cond_mean, cond_cov)。

    用分块矩阵公式（讲义 §1.2 的推广，Schur 补）：
        Sigma = [[A, B], [B^T, C]]   A 对应观测维，C 对应剩余维
        cond_mean = mu_rest + B^T A^{-1} (x_obs - mu_obs)
        cond_cov  = C - B^T A^{-1} B
    线性方程组用 mx.linalg.solve（stream=mx.cpu，linalg 不支持 GPU），禁止显式求逆。
    """
    raise NotImplementedError("TODO(lab02 §1.2): 分块 + Schur 补")


def gaussian_mixture_sample(weights, mus, Sigmas, n: int) -> mx.array:
    """从混合高斯采样：先按 weights 抽成分，再从对应高斯采样。

    MLX 无 rng.choice：采样 u ~ mx.random.uniform((n,))，与累积权重
    （mx.cumsum）比较决定每个样本的成分。
    用于讲义 §1.3 反例 B：weights=[0.5,0.5], Sigmas=[Sigma_p, Sigma_m]。
    """
    raise NotImplementedError("TODO(lab02 §1.3): 两段式采样")


def empirical_conditional(samples, x0: float, delta: float, obs_idx: int = 0):
    """经验条件分布（讲义 §1.4）：取窄带 |samples[:,obs_idx] - x0| <= delta 内的样本。
    MX 不支持布尔索引：取子数组用 NumPy 桥接
    mask = np.array(mx.abs(...) <= delta)；sub = mx.array(np.array(samples)[mask])。
    返回 (其余维度样本, 带内样本数)。"""
    raise NotImplementedError("TODO(lab02 §1.4): 窄带筛选")
