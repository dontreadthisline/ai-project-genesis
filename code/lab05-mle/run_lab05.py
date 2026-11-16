"""Lab 05 入口：最大似然估计。

流程：check_implementation()（讲义 T1-T3）-> 对照实验 E1-E5（讲义 §3）-> 产出图表与 metrics.json。
运行：uv run run_lab05.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimators import gaussian_map, gaussian_mle, gaussian_nll

OUTPUT = Path(__file__).parent / "output"
FIGURES = OUTPUT / "figures"

TRUE_MU, TRUE_SIGMA2 = 3.0, 4.0


# ---------------------------------------------------------------- 自测（讲义 §2）


def check_implementation() -> None:
    rng = np.random.default_rng(0)

    # T1: 闭式解确实是 NLL 最小值 —— 网格搜索对照
    x = rng.normal(1.5, 2.0, size=500)
    mu_hat, s2_hat = gaussian_mle(x)
    mus = np.linspace(1.2, 1.8, 81)
    s2s = np.linspace(3.4, 4.4, 101)
    grid = np.array([[gaussian_nll((m, s), x) for s in s2s] for m in mus])
    i, j = np.unravel_index(grid.argmin(), grid.shape)
    assert abs(mus[i] - mu_hat) < 0.01 and abs(s2s[j] - s2_hat) < 0.01

    # T2: 精度对照（NumPy 默认 ddof=0）
    assert abs(mu_hat - x.mean()) < 1e-12
    assert abs(s2_hat - x.var()) < 1e-12

    # T3: MAP 收敛检查：n 大时 MAP -> MLE；强先验时 MAP -> mu0
    big = rng.normal(5.0, 1.0, size=100_000)
    assert abs(gaussian_map(big, mu0=0.0, tau2=1.0, sigma2=1.0) - big.mean()) < 1e-3
    assert abs(gaussian_map(x[:5], mu0=0.0, tau2=1e-6, sigma2=4.0)) < 1e-3

    print("[check] T1-T3 全部通过")


# ---------------------------------------------------------------- 对照实验（讲义 §3）


def E1_sampling_distributions(metrics: dict) -> None:
    """E1：n in {5,10,30,100,1000,1e4}，每档重复 2000 次。

    六联直方图（mu_hat 一行、sigma2_hat 一行，讲义图 1）。
    metrics['E1_mu_dist'] 与 metrics['E1_sigma2_dist'] 记录各档 mean/sd/bias。
    """
    raise NotImplementedError("TODO(lab05 E1)")


def E2_sigma2_bias(metrics: dict) -> None:
    """E2：1/n 版 vs 1/(n-1) 版 sigma2 的经验偏差曲线 + 理论线 -sigma^2/n（讲义图 2）。

    metrics['E2_bias_curve'] = {"n": [...], "bias_1_over_n": [...], "bias_1_over_n_minus_1": [...]}
    """
    raise NotImplementedError("TODO(lab05 E2)")


def E3_clt_check(metrics: dict) -> None:
    """E3：mu_hat 经验标准差 vs 理论 sigma/sqrt(n)（对数-对数，斜率 -1/2，讲义图 3）。

    metrics['E3_sd_vs_theory'] = {"n10": {"emp":..., "theo":...}, ...}
    """
    raise NotImplementedError("TODO(lab05 E3)")


def E4_map_shrinkage(metrics: dict) -> None:
    """E4：tau2 in {1e2, 1, 1e-2}，n in {5, 50, 500}，mu0=0（偏离真值 3）。

    MAP 相对 MLE 的收缩量（讲义图 4）。验证：n 大时 MAP->MLE，强先验+小样本时 MAP->0。
    """
    raise NotImplementedError("TODO(lab05 E4)")


def E5_misspecification(metrics: dict) -> None:
    """E5：数据改从 t3（rng.standard_t(3, n)*np.sqrt(4/3)，校准方差 4），拟合仍是高斯 MLE。

    对比高斯数据与 t 数据下 sigma2_hat 的抽样分布（讲义图 5）：
    metrics['E5_misspec'] = {"gaussian_data_mean_s2hat":..., "t_data_mean_s2hat":...}
    """
    raise NotImplementedError("TODO(lab05 E5)")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    check_implementation()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"T_all_passed": True}

    E1_sampling_distributions(metrics)
    E2_sigma2_bias(metrics)
    E3_clt_check(metrics)
    E4_map_shrinkage(metrics)
    E5_misspecification(metrics)

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] 图表 -> {FIGURES}，指标 -> {OUTPUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
