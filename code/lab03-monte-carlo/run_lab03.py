"""Lab 03 入口：Monte Carlo 估计。

流程：check_implementation()（讲义 T1-T5）-> 对照实验 E1-E4（讲义 §3）-> 产出图表与 metrics.json。
运行：uv run run_lab03.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlx.core as mx

from estimators import mc_estimate

OUTPUT = Path(__file__).parent / "output"
FIGURES = OUTPUT / "figures"

STD_NORMAL = lambda n: mx.random.normal((n,))
UNIT_SQUARE = lambda n: mx.random.uniform(-1, 1, shape=(n, 2))


# ---------------------------------------------------------------- 自测（讲义 §2）


def check_implementation() -> None:
    import math

    mx.set_default_device(mx.cpu)   # 本阶段：fp64 检验与 linalg 均需 CPU
    mx.random.seed(0)

    # T1: E[X] = 0, E[X^2] = 1（n = 1e6，2 个标准误差内）
    r1 = mc_estimate(lambda z: z, STD_NORMAL, 10**6)
    assert abs(r1["est"] - 0.0) < 2 * r1["se"]
    r2 = mc_estimate(lambda z: z**2, STD_NORMAL, 10**6)
    assert abs(r2["est"] - 1.0) < 2 * r2["se"]

    # T3: E[|X|] = sqrt(2/pi)
    r3 = mc_estimate(lambda z: mx.abs(z), STD_NORMAL, 10**6)
    assert abs(r3["est"] - math.sqrt(2 / math.pi)) < 2 * r3["se"]

    # T4: E[sin X] = 0（奇函数，对称分布）
    r4 = mc_estimate(lambda z: mx.sin(z), STD_NORMAL, 10**6)
    assert abs(r4["est"]) < 2 * r4["se"]

    # T5: pi/4 估计
    r5 = mc_estimate(lambda P: (P**2).sum(axis=1) <= 1, UNIT_SQUARE, 10**6)
    assert abs(r5["est"] - math.pi / 4) < 5e-3

    print("[check] T1-T5 全部通过")


# ---------------------------------------------------------------- 对照实验（讲义 §3）


def E1_convergence(metrics: dict) -> None:
    """E1：n in {1e1..1e6}，f(x)=x，每档重复 100 次取平均绝对误差。

    双对数图 + 理论线 se = sigma_f/sqrt(n)，拟合斜率写入 metrics['E1_loglog_slope']
    （验收要求落在 [-0.55, -0.45]）。
    """
    raise NotImplementedError("TODO(lab03 E1): 讲义图 1 规格")


def E2_function_hardness(metrics: dict) -> None:
    """E2：f in {x+3, x^2, |x|, 1[x>3]}，n=1e5，重复 1000 次。

    每个函数记录经验标准差、理论 se、相对误差：
    metrics['E2_per_function'] = {"x+3": {"emp_sd":..., "theo_se":..., "rel_err":...}, ...}
    四联直方图 + 正态曲线叠加（CLT 验证，讲义图 2）。
    """
    raise NotImplementedError("TODO(lab03 E2)")


def E3_rare_events(metrics: dict) -> None:
    """E3：f = 1[x>c]，c in {1,2,3,4,5}，n=1e6，重复 200 次。

    对数-对数图：p_c vs 相对误差 + 理论线 1/sqrt(n*p_c)（讲义图 3）。
    metrics['E3_rare_event'] = {"c=1": {"p_hat":..., "rel_err":...}, ...}
    """
    raise NotImplementedError("TODO(lab03 E3)")


def E4_pi(metrics: dict) -> None:
    """E4：pi 估计，n 从 1e2 到 1e7，误差收敛曲线 + 1/sqrt(n) 参考线（讲义图 4）。"""
    raise NotImplementedError("TODO(lab03 E4)")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    check_implementation()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"T_all_passed": True}

    E1_convergence(metrics)
    E2_function_hardness(metrics)
    E3_rare_events(metrics)
    E4_pi(metrics)

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] 图表 -> {FIGURES}，指标 -> {OUTPUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
