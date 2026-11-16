---
title: "第一阶段：概率建模与信息论基础"
---

核心问题：如何使用数学语言描述随机现象、概率分布，以及不同概率分布之间的关系？

本阶段是后续 VAE、Diffusion、RL 和 LLM 后训练的共同数学基础。概念复用地图：

- Lab 03 的 Monte Carlo 估计 → 策略梯度估计（实验 28/30）、ELBO 采样估计（实验 15）
- Lab 04 的 KL 散度 → VAE 正则项（实验 15）、PPO KL 约束（实验 30）、DPO（实验 39）
- Lab 05 的 MLE = 最小化交叉熵 → 所有深度学习分类/语言模型训练（实验 6、35）
- Lab 02 的条件分布 → 扩散模型的条件分布链（实验 19）、RL 的转移核

## 讲义列表

| 讲义 | 实验 | 主题 |
|---|---|---|
| [lab01-pdf-and-covariance.md](lab01-pdf-and-covariance.md) | 实验 1 | PDF 与采样直方图、均值方差的几何意义、协方差矩阵 |
| [lab02-joint-and-conditional.md](lab02-joint-and-conditional.md) | 实验 2 | 联合/边缘/条件分布、相同边缘不同联合 |
| [lab03-monte-carlo.md](lab03-monte-carlo.md) | 实验 3 | Monte Carlo 估计、标准误差、收敛速率 |
| [lab04-entropy-kl.md](lab04-entropy-kl.md) | 实验 4 | Entropy / Cross Entropy / KL / Brier Score |
| [lab05-mle.md](lab05-mle.md) | 实验 5 | 最大似然估计、估计偏差、与交叉熵的等价性 |

每个讲义固定结构：理论问题 → 数学基础（含推导）→ 实现任务（含代码接口与自测）→ 对照实验 → 预期图表 → 观测指标 → 结果分析 → 验收清单。

## 环境搭建

使用 uv 管理环境：

```bash
uv init genesis-stage1
cd genesis-stage1
uv add numpy matplotlib
uv add mlx   # 可选：Apple Silicon 上的数组后端练习
```

统一随机数约定：所有实验使用 `rng = np.random.default_rng(seed)`，默认 `seed = 0`，保证结果可复现。

## 代码布局（code/）

每个实验是一个独立的 uv 项目（独立 pyproject.toml 与 .venv），依赖仅 numpy 与 matplotlib：

```
code/
├── lab01-pdf-and-covariance/    # distributions.py / viz.py / run_lab01.py
├── lab02-joint-and-conditional/  # distributions.py（含 lab01 复制项）/ run_lab02.py
├── lab03-monte-carlo/            # estimators.py / run_lab03.py
├── lab04-entropy-kl/             # metrics.py / run_lab04.py
└── lab05-mle/                    # distributions.py（复制项）/ estimators.py / run_lab05.py
```

每个项目的运行入口 `run_labXX.py` 固定流程：自测（讲义 TX 断言）→ 对照实验（讲义 EX）→ 产出 `output/figures/*.png` 与 `output/metrics.json`（观测指标 JSON 落盘，供阶段验收检查）。自测未通过时直接失败，不产出实验结果。

模块内 TODO 骨架均标注对应讲义章节；lab02/lab05 需要复用 lab01 完成的函数，模块内注释已标明复制来源。统一命令：

```bash
cd code/lab01-pdf-and-covariance
uv sync
uv run run_lab01.py
```

## 阶段验收标准

能独立解释和计算：

- [ ] 联合分布、边缘分布、条件分布之间的关系（能手工推导二维高斯条件分布公式）
- [ ] 随机向量的均值与协方差（能从协方差矩阵预测等密度椭圆的形状与朝向）
- [ ] 数学期望的 Monte Carlo 估计（能解释标准误差、1/√n 收敛、CLT 近似）
- [ ] MLE、Cross Entropy 和 KL 的联系（能推导 H(P,Q) = H(P) + KL(P‖Q) 与 NLL 均值 = 交叉熵）
- [ ] 五个实验项目全部自测变绿，run_labXX.py 可独立运行并产出图表与 metrics.json

## MLX 练习（可选）

完成 NumPy 版本后，用 `mx.array` 重写 `distributions.py` 与 `metrics.py`（MLX 数组 API 与 NumPy 高度兼容），用 `mx.eval` 触发惰性求值，并与 NumPy 结果逐项对拍（容差 1e-10）。此练习的目的仅是熟悉数组后端切换，本阶段不涉及自动微分。
