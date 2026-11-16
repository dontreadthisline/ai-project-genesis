"""Lab 04 入口：概率分布度量。

流程：check_implementation()（讲义 T1-T6）-> 对照实验 E1-E5（讲义 §3）-> 产出图表与 metrics.json。
运行：uv run run_lab04.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import brier_score, cross_entropy, entropy, kl_divergence

OUTPUT = Path(__file__).parent / "output"
FIGURES = OUTPUT / "figures"


# ---------------------------------------------------------------- 自测（讲义 §2）


def check_implementation() -> None:
    rng = np.random.default_rng(0)

    # T1: 恒等式 H(P,Q) = H(P) + KL(P||Q)
    p = rng.dirichlet([1] * 5)
    q = rng.dirichlet([1] * 5)
    lhs = cross_entropy(p, q)
    rhs = entropy(p) + kl_divergence(p, q)
    assert abs(lhs - rhs) < 1e-10

    # T2: KL 非负，且 KL(P||P)=0
    assert kl_divergence(p, p) == 0.0
    assert kl_divergence(p, q) >= 0.0

    # T3: 均匀分布熵 = log K
    u = np.ones(5) / 5
    assert abs(entropy(u) - np.log(5)) < 1e-12

    # T4: 确定性分布熵 = 0
    assert entropy(np.array([0.0, 1.0, 0.0])) == 0.0

    # T5: 支撑违规 -> inf
    p_bad = np.array([0.5, 0.5, 0.0])
    q_bad = np.array([0.7, 0.0, 0.3])
    assert np.isinf(cross_entropy(p_bad, q_bad))

    # T6: 手算对照（讲义 §1.3）
    P = np.array([0.5, 0.5])
    Q = np.array([0.99, 0.01])
    assert abs(kl_divergence(P, Q) - 1.6144) < 5e-3
    assert abs(kl_divergence(Q, P) - 0.6372) < 5e-3

    print("[check] T1-T6 全部通过")


# ---------------------------------------------------------------- 对照实验（讲义 §3）


def E1_scan_q(metrics: dict) -> None:
    """E1：P=(0.7,0.3) 固定，Q=(t,1-t) 扫 [0.001,0.999]。

    三条曲线 H(P,Q)、D_KL、H(P) 水平线；最小值位置写入
    metrics['E1_min_positions']（两个 argmin 都应为 0.7，H(P,Q) 最小值应落在 H(P) 上）。
    """
    raise NotImplementedError("TODO(lab04 E1): 讲义图 1 规格")


def E2_asymmetry(metrics: dict) -> None:
    """E2：20 对随机分布（rng.dirichlet），散点图 D(P||Q) vs D(Q||P) + 对角线。

    metrics['E2_asymmetry'] = {"max_abs_diff":..., "mean_forward":..., "mean_backward":...}
    """
    raise NotImplementedError("TODO(lab04 E2): 讲义图 2 规格")


def E3_proper_scoring(metrics: dict) -> None:
    """E3：真实 p=0.7 的伯努利，预测 q 扫 [0.01,0.99]，每格点 1e5 个 MC 样本。

    期望 CE 与期望 Brier 曲线（讲义图 3），最小值位置写入 metrics['E3_proper_check']
    （两者都应接近 0.7）。期望 CE 用样本平均 -log q(y) 计算（即经验 NLL）。
    """
    raise NotImplementedError("TODO(lab04 E3)")


def E4_ce_vs_brier(metrics: dict) -> None:
    """E4：q_y in {1, 0.9, 0.5, 0.1, 0.01, 1e-6} 的损失对照柱状图。

    metrics['E4_table'] = {"qy_1": {"CE":..., "Brier":...}, ..., "qy_1e-6": {"CE":..., "Brier":...}}
    """
    raise NotImplementedError("TODO(lab04 E4): 讲义图 4 规格")


def E5_gaussian_kl(metrics: dict) -> None:
    """E5：高斯对高斯 KL 热力图（离散格点近似）vs 闭式解（讲义 §1.5）。

    格点 [-8,8]、delta=0.01 离散化两条密度并归一化，(mu2, sigma2) 网格上画离散 KL，
    叠加闭式解等高线；最大相对误差写入 metrics['E5_gaussian_kl_max_rel_err']。
    """
    raise NotImplementedError("TODO(lab04 E5): 讲义图 5 规格")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    check_implementation()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"T_all_passed": True}

    E1_scan_q(metrics)
    E2_asymmetry(metrics)
    E3_proper_scoring(metrics)
    E4_ce_vs_brier(metrics)
    E5_gaussian_kl(metrics)

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] 图表 -> {FIGURES}，指标 -> {OUTPUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
