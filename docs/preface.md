---
title: "前言"
---

这是一门以实验为骨架的课程。数学推导服务于实验，知识点服务于实验，最终目标只有一个：独立实现并训练一个 Agentic RL 系统。

## 主线

$$
\text{Probability}
\rightarrow
\text{Deep Learning}
\rightarrow
\text{Generative Models}
\rightarrow
\text{RL}
\rightarrow
\text{LLM}
\rightarrow
\text{Agentic RL}
$$

六条线里有三条主要实验脉络：MNIST（深度学习与生成模型）、Gymnasium（强化学习）、TinyStories（语言模型）。Python 与 MLX 不单独设阶段，贯穿所有项目。

## 本书结构

全书六个部分对应课程六个阶段，每个阶段围绕一个明确的研究问题展开：

| 部分 | 阶段 | 核心问题 | 实验脉络 |
|---|---|---|---|
| 一 | 概率建模与信息论 | 如何用数学语言描述随机现象与分布之间的关系？ | MLX 分布实验 |
| 二 | 深度学习与序列建模 | 神经网络如何通过梯度优化学习复杂函数？ | MNIST |
| 三 | 生成模型 | 如何让网络学习数据分布并生成新样本？ | MNIST |
| 四 | 强化学习 | 通过与环境交互获得反馈时如何优化策略？ | Gymnasium |
| 五 | LLM 预训练与后训练 | 深度学习与 RL 机制如何应用于自回归语言模型？ | TinyStories |
| 六 | Agentic RL | 语言模型如何通过多步环境交互完成多步任务？ | 可验证工具环境 |

每个实验固定结构：理论问题、数学基础（含完整推导）、实现任务（代码接口与自测）、对照实验、预期图表、观测指标、结果分析、验收清单。同一数学概念会在多个阶段反复出现（Monte Carlo → 策略梯度；KL → VAE/PPO/DPO；MLE = 交叉熵 → 一切训练），各章末尾的「概念接口」标注这些锚点。

## 实现约定：MLX 优先

全书实验默认用 **MLX** 实现数组计算、模型与训练（`import mlx.core as mx`；神经网络用 `mlx.nn`）。NumPy 不再作为实现选项，仅限以下边界场景使用：

- **Matplotlib 桥接**：画图前 `np.array(mx_array)` 转换（matplotlib 不认 mlx 数组）；
- **数据边界**：文件下载与字节解析（如 MNIST idx 格式）、Gymnasium 观测的搬运；
- **缺失原语的明确标注**：个别统计原语 MLX 暂缺时，讲义会明确标注「NumPy 边界」及原因，不允许默默回退。

本环境实测（mlx 0.32）的三个已知局限及标准应对，全书统一遵守：

- **fp64 不支持 GPU**。本阶段需要 fp64 做数值梯度检验与闭式对照，脚本开头统一 `mx.set_default_device(mx.cpu)`；第一阶段全是小规模统计计算，CPU 没有性能损失。GPU 从第二阶段大规模张量运算（全 fp32）开始使用。
- **`mx.linalg` 四件套（`slogdet / solve / cholesky / eigh`）不支持 GPU**，必须显式传 `stream=mx.cpu`，否则抛错。
- **布尔索引不支持**（`a[mask]` 报错）。两类替代：求和场景用 `mx.sum(mx.where(mask, vals, 0.0))`；取子数组场景用 NumPy 桥接 `mx.array(np.array(a)[np.array(mask)])`（这是合法的「NumPy 边界」用法）。

随机数约定：每个实验脚本开头 `mx.random.seed(0)`（全局 PRNG），函数签名不携带随机状态；需要严格函数式复现时改用 `mx.random.key()` / `mx.random.split()`。惰性求值提醒：打印、比较、绘图前需要 `mx.eval()` 强制求值。

## 如何使用本书

本书与代码仓库是同一个课程的两条轨道，配套使用：

- **讲义（本书）**：阅读推导、明确实验设计、对照验收清单。
- **代码（`code/`）**：每个实验一个独立的 uv 项目，TODO 骨架加自测红灯。先让自测变绿，再做对照实验，产出图表与 `metrics.json`。

```bash
cd code/lab01-pdf-and-covariance
uv sync                 # 安装依赖
uv run run_lab01.py     # 自测 -> 对照实验 -> output/
```

时间不是硬性进度要求：每阶段必须达到验收标准，才能进入下一阶段。每周 10 小时的初始估算下，全程约 5-7 个月。

## 当前进度

- 第一阶段：五个实验的讲义与代码骨架已就绪。
- 第二至第六阶段：阶段导览已建，实验讲义按大纲（实验 6-45）逐步展开，每完成一个实验即加入对应部分。

## 内容与渲染分离

本书内容源（`docs/`）与渲染构建（Quarto，经 `scripts/build-book.sh`）是分离的：书籍结构以 [manifest.yml](manifest.yml) 为单一事实源描述，渲染器只消费该清单。未来切换其他构建工具（VitePress、mdBook 等）时，只需为新工具编写一个清单生成器，内容文件不需要任何改动。
