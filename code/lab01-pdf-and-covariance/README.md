# Lab 01：一维与二维概率分布

讲义：[../../docs/stage1-probability/lab01-pdf-and-covariance.md](../../docs/stage1-probability/lab01-pdf-and-covariance.md)

## 任务

实现用 MLX（`mx.array`，随机数用 `mx.random.seed(0)` 全局种子）；NumPy 仅作 matplotlib 桥接（画图前 `np.array(mx_arr)`）。

实现 `distributions.py` 中的三个函数（TODO 骨架，公式见讲义 §1.2 / §1.3 / §2），然后完成 `run_lab01.py` 中的对照实验 E1-E6（讲义 §3）。

## 命令

```bash
uv sync                 # 安装依赖
uv run run_lab01.py     # 自测 T1-T4 -> 对照实验 -> output/figures/*.png + output/metrics.json
```

自测未通过时脚本直接失败，不产出实验结果（红灯-绿灯工作流：先让 T1-T4 变绿，再做实验）。
