# Lab 02：联合分布与条件分布

讲义：[../../docs/stage1-probability/lab02-joint-and-conditional.md](../../docs/stage1-probability/lab02-joint-and-conditional.md)

## 任务

实现用 MLX（`mx.array`，随机数用 `mx.random.seed(0)` 全局种子）；NumPy 仅作 matplotlib 桥接（画图前 `np.array(mx_arr)`）。

1. 把 lab01 完成的三个分布函数复制进 `distributions.py`（文件内已标注）。
2. 实现新增函数：`gaussian_conditional`（Schur 补）、`gaussian_mixture_sample`、`empirical_conditional`。
3. 完成 `run_lab02.py` 中的对照实验 E1-E4（讲义 §3）。

## 命令

```bash
uv sync
uv run run_lab02.py     # 自测 T1-T3 -> 对照实验 -> output/
```
