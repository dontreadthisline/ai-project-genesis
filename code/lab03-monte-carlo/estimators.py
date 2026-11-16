"""Lab 03 估计器模块：Monte Carlo 估计。

实现要求与公式见 docs/stage1-probability/lab03-monte-carlo.md（§1.1-§1.2、§2）。
"""
import numpy as np


def mc_estimate(f, sampler, n: int, rng) -> dict:
    """Monte Carlo 估计 E[f(X)]（讲义 §1.1/§1.2）。

    参数：
        f        : 向量化函数 np.ndarray -> np.ndarray
        sampler  : callable(n, rng) -> np.ndarray，从目标分布采样 n 个点
        n        : 样本量
        rng      : np.random.Generator

    返回 {"est": float, "se": float, "n": int}：
        est = mean(f(X_i))
        se  = 样本标准差 / sqrt(n)，样本方差必须用 1/(n-1)（讲义 §1.2、§6 问题 4）
    """
    raise NotImplementedError("TODO(lab03 §1.2): 注意 1/(n-1)")
