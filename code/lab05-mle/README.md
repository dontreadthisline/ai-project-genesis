# Lab 05：最大似然估计

讲义：[../../docs/stage1-probability/lab05-mle.md](../../docs/stage1-probability/lab05-mle.md)

## 任务

实现用 MLX（`mx.array`，随机数用 `mx.random.seed(0)` 全局种子）；NumPy 仅作 matplotlib 桥接（画图前 `np.array(mx_arr)`）。

1. 把 lab01 完成的 `univariate_gaussian_pdf` 复制进 `distributions.py`（文件内已标注）。
2. 实现 `estimators.py` 中的 `gaussian_nll` / `gaussian_mle` / `gaussian_map`。
3. 完成 `run_lab05.py` 中的对照实验 E1-E5（讲义 §3）。

## 命令

```bash
uv sync
uv run run_lab05.py     # 自测 T1-T3 -> 对照实验 -> output/
```
