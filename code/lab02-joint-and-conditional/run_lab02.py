"""Lab 02 入口：联合分布与条件分布。

流程：check_implementation()（讲义 T1-T3）-> 对照实验 E1-E4（讲义 §3）-> 产出图表与 metrics.json。
运行：uv run run_lab02.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np   # 仅画图桥接：np.array(mx_arr)

import mlx.core as mx

from distributions import (
    empirical_conditional,
    gaussian_conditional,
    gaussian_mixture_sample,
    sample_multivariate_gaussian,
)

OUTPUT = Path(__file__).parent / "output"
FIGURES = OUTPUT / "figures"


# ---------------------------------------------------------------- 自测（讲义 §2）


def sample_cov(X):
    """样本协方差（复制自 lab01，ddof=1）。"""
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def check_implementation() -> None:
    mx.set_default_device(mx.cpu)   # 本阶段：fp64 检验与 linalg 均需 CPU
    mx.random.seed(0)

    # T1: 与 §1.2 手推公式对照（rho=0.6, sx=1.5, sy=0.8, x0=1.2；fp64）
    mu = mx.zeros((2,))
    sx, sy, rho = 1.5, 0.8, 0.6
    Sigma = mx.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]], dtype=mx.float64)
    m, S = gaussian_conditional(mu, Sigma, x_obs=1.2, obs_idx=0)
    assert abs(m[0].item() - rho * sy / sx * 1.2) < 1e-12
    assert abs(S[0, 0].item() - sy**2 * (1 - rho**2)) < 1e-12

    # T2: rho -> 0 时条件分布退化为边缘
    Sigma0 = mx.array([[1.0, 0.0], [0.0, 2.0]], dtype=mx.float64)
    m0, S0 = gaussian_conditional(mx.zeros((2,)), Sigma0, x_obs=3.0, obs_idx=0)
    assert abs(m0[0].item()) < 1e-12 and abs(S0[0, 0].item() - 2.0) < 1e-12

    # T3: 混合高斯边缘校验（反例 B）：边缘均值 0、方差 1，整体协方差近似为零
    Sigma_p = mx.array([[1.0, 0.9], [0.9, 1.0]])
    Sigma_m = mx.array([[1.0, -0.9], [-0.9, 1.0]])
    XY = gaussian_mixture_sample([0.5, 0.5], [mx.zeros((2,))] * 2, [Sigma_p, Sigma_m], 400_000)
    mx.eval(XY)
    assert abs(XY[:, 0].mean().item()) < 0.01 and abs(XY[:, 0].var().item() - 1.0) < 0.02
    assert abs(sample_cov(XY)[0, 1].item()) < 0.02

    print("[check] T1-T3 全部通过")


# ---------------------------------------------------------------- 对照实验（讲义 §3）


def E1_rho_sweep(metrics: dict) -> None:
    """E1：rho in {-0.9,-0.5,0,0.5,0.9}，n=1e5。五联图：联合散点 + 两侧边缘直方图。

    预期：边缘全部近似 N(0,1)（写入 metrics['E1_marginal_check']），联合形态随 rho 变化。
    """
    raise NotImplementedError("TODO(lab02 E1): 讲义图 1 规格")


def E2_conditional_at_x0(metrics: dict) -> None:
    """E2：rho=0.8，x0 in {-1.5, 0, 1.5}，窄带 delta=0.05。

    每个格点记录经验条件均值/标准差 vs 理论值（gaussian_conditional）：
    metrics['E2_conditional_check'] = {"x0_-1.5": {"emp_mean":..., "theo_mean":..., "emp_std":..., "theo_std":...}, ...}
    """
    raise NotImplementedError("TODO(lab02 E2): 讲义图 2 规格（条件直方图 + 理论密度 + 边缘密度虚线）")


def E3_band_tradeoff(metrics: dict) -> None:
    """E3：delta in {0.01, 0.05, 0.2, 0.5}，x0=1，rho=0.8，n=1e5（再对照 n=1e4）。

    记录各 delta 的条件均值估计偏差与带内样本数：
    metrics['E3_band_tradeoff'] = {"delta_0.01": {"bias":..., "n_in_band":...}, ...}
    """
    raise NotImplementedError("TODO(lab02 E3): 双 y 轴图，讲义图 3 规格")


def E4_same_marginals(metrics: dict) -> None:
    """E4：三种联合（rho=0.8 高斯 / rho=0 高斯 / 混合反例 B），边缘全部 N(0,1)。

    记录三者经验协方差与 x0=1.5 处条件分布形状（混合情形应呈双峰）：
    metrics['E4_cov'] = {...}
    """
    raise NotImplementedError("TODO(lab02 E4): 三联图，讲义图 4 规格")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    check_implementation()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"T_all_passed": True}

    E1_rho_sweep(metrics)
    E2_conditional_at_x0(metrics)
    E3_band_tradeoff(metrics)
    E4_same_marginals(metrics)

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] 图表 -> {FIGURES}，指标 -> {OUTPUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
