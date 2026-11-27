# Lab 15

讲义：`docs/stage3-generative-models/lab15-vae.md`

- `skeleton.py` 接口骨架（TODO 即实现要求）
- `selftest.py` 自测断言（先让它们变绿）
- `run_lab{num:02d}.py` 实验入口（自测 -> 对照实验 E 系列 -> output/）

```bash
uv sync && uv run run_lab{num:02d}.py
```

观测指标落盘约定: 对照实验因变量写入 `output/metrics.json`（键名与讲义 §3/§5 一致），图表写入 `output/figures/`。
