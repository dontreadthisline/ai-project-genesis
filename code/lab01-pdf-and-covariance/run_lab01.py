"""Lab 01 入口：一维与二维概率分布。

流程：check_implementation()（讲义 T1-T4）-> 对照实验 E1-E6（讲义 §3）-> 产出图表与 metrics.json。
运行：uv run run_lab01.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np   # 仅画图桥接：np.array(mx_arr)

import mlx.core as mx

from distributions import (
    multivariate_gaussian_pdf,
    sample_multivariate_gaussian,
    univariate_gaussian_pdf,
)
from viz import plot_density_ellipses, plot_pdf_and_hist

OUTPUT = Path(__file__).parent / "output"
FIGURES = OUTPUT / "figures"


# ---------------------------------------------------------------- 自测（讲义 §2）


def trapezoid(y, x):
    """梯形积分（MLX 无 np.trapezoid 对应物，手写）。"""
    return mx.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)


def sample_cov(X):
    """样本协方差（MLX 无 cov，手写，ddof=1）。"""
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def check_implementation() -> None:
    mx.set_default_device(mx.cpu)   # 本阶段：fp64 检验与 linalg 均需 CPU
    mx.random.seed(0)

    # T1: 数值积分验证归一化（fp64，容差 1e-8）
    x = mx.linspace(-12, 12, 200_001, dtype=mx.float64)
    y = univariate_gaussian_pdf(x, 2.0, 3.0)
    assert abs(trapezoid(y, x).item() - 1.0) < 1e-8

    # T2: 二维密度在均值处取最大值
    Sigma = mx.array([[2.0, 0.6], [0.6, 1.0]])
    mu2 = mx.array([1.0, -1.0])
    pts = mx.random.normal((5000, 2)) @ mx.linalg.cholesky(Sigma, stream=mx.cpu) + mu2
    vals = multivariate_gaussian_pdf(pts, mu2, Sigma)
    mx.eval(vals)
    assert (
        multivariate_gaussian_pdf(mu2[None, :], mu2, Sigma).item()
        >= vals.max().item() - 1e-12
    )

    # T3: 采样协方差收敛于 Sigma（n=200_000，逐项容差 0.05）
    X = sample_multivariate_gaussian(mu2, Sigma, 200_000)
    mx.eval(X)
    assert mx.allclose(X.mean(axis=0), mu2, atol=0.03)
    assert mx.allclose(sample_cov(X), Sigma, atol=0.05)

    # T4: 标准正态在 0 处密度 = 1/sqrt(2*pi)
    assert abs(
        univariate_gaussian_pdf(mx.array(0.0), 0.0, 1.0).item()
        - 1 / math.sqrt(2 * math.pi)
    ) < 1e-12

    print("[check] T1-T4 全部通过")


# ---------------------------------------------------------------- 对照实验（讲义 §3）


def E1_mu_shift(metrics: dict) -> None:
    """E1：mu in {-2, 0, 3}，sigma=1。产出图 1（与 E2 合并为一张多子图）。"""
    raise NotImplementedError("TODO(lab01 E1): 讲义 §3 表 + 图 1 规格")


def E2_sigma_width(metrics: dict) -> None:
    """E2：sigma in {0.2, 1, 5}，mu=0。记录各 sigma 的密度峰值，验证峰高 ∝ 1/sigma。"""
    raise NotImplementedError("TODO(lab01 E2): 峰值写入 metrics['E2_peak_heights']")


def E3_sample_size(metrics: dict) -> None:
    """E3：n in {100, 1e3, 1e4, 1e5}，直方图 vs PDF 的最大偏差。

    metrics['E3_max_hist_pdf_deviation'] = {"n100": ..., "n1e3": ..., "n1e4": ..., "n1e5": ...}
    偏差定义：bin 中心处 max_b |p_hat_b - p(x_b)|。
    """
    raise NotImplementedError("TODO(lab01 E3): 四联图 + 偏差统计")


def E4_bin_width(metrics: dict) -> None:
    """E4：同一份 n=1e4 样本，bins in {5, 20, 50, 200}。观察过平滑与抖动。"""
    raise NotImplementedError("TODO(lab01 E4)")


def _bivariate_sigma(rho: float) -> np.ndarray:
    return np.array([[1.0, rho], [rho, 1.0]])


def E5_correlation(metrics: dict) -> None:
    """E5：rho in {-0.9, -0.5, 0, 0.5, 0.9}。五行两列：样本云 + 等密度椭圆（两层 c=1,2）。

    每行记录轴比 metrics['E5_ellipse_axis_ratio'][f'rho_{rho}'] = sqrt(l1/l2)，
    以及经验协方差 metrics['E5_sample_cov']。
    """
    raise NotImplementedError("TODO(lab01 E5): 用 viz.plot_density_ellipses")


def E6_anisotropic(metrics: dict) -> None:
    """E6：Sigma=diag(9,1)、diag(1,9)、旋转 45 度版。验证轴比 3:1 与倾角。"""
    raise NotImplementedError("TODO(lab01 E6)")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    check_implementation()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"T_all_passed": True}

    E1_mu_shift(metrics)
    E2_sigma_width(metrics)
    E3_sample_size(metrics)
    E4_bin_width(metrics)
    E5_correlation(metrics)
    E6_anisotropic(metrics)

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] 图表 -> {FIGURES}，指标 -> {OUTPUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
