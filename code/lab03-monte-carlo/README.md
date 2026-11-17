# Lab 03：Monte Carlo 估计

讲义：[../../docs/stage1-probability/lab03-monte-carlo.md](../../docs/stage1-probability/lab03-monte-carlo.md)

## 任务

实现用 MLX（`mx.array`，随机数用 `mx.random.seed(0)` 全局种子）；NumPy 仅作 matplotlib 桥接（画图前 `np.array(mx_arr)`）。

实现 `estimators.py` 中的 `mc_estimate`，然后完成 `run_lab03.py` 中的对照实验 E1-E4（讲义 §3）。

本实验无前置依赖（不依赖 lab01/lab02 的模块）。

## 命令

```bash
uv sync
uv run run_lab03.py     # 自测 T1-T5 -> 对照实验 -> output/
```
