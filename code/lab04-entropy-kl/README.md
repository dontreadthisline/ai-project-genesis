# Lab 04：概率分布度量

讲义：[../../docs/stage1-probability/lab04-entropy-kl.md](../../docs/stage1-probability/lab04-entropy-kl.md)

## 任务

实现用 MLX（`mx.array`，随机数用 `mx.random.seed(0)` 全局种子）；Matplotlib 对 mx.array 零桥接直接可用，numpy 仅布尔索引取子数组时导入。

实现 `metrics.py` 中的四个函数，然后完成 `run_lab04.py` 中的对照实验 E1-E5（讲义 §3）。

## 命令

```bash
uv sync
uv run run_lab04.py     # 自测 T1-T6 -> 对照实验 -> output/
```
