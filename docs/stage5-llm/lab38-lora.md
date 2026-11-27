---
title: "Lab 38：LoRA 对照"
lab: 38
stage: 5
---

对应大纲实验 38。依赖 Lab 37 的 SFT 任务。

## 0. 理论问题

1. 低秩假设「微调的增量是低秩的」直觉上为什么成立？
2. LoRA 的前向改了什么、梯度流向哪？
3. 参数效率的代价在什么场景显现？

## 1. 数学基础

### 1.1 LoRA 参数化

冻结 $W_0 \in \mathbb R^{out \times in}$，可训练增量分解：

$$
h = W_0 x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb R^{r \times in},\ B \in \mathbb R^{out \times r},\ r \ll \min(in, out)
$$

初始化 $A \sim \mathcal N(0, \sigma^2)$、$B = 0$：初始增量严格为零，训练从原模型精确出发（对照 Lab 11 残差零初始化的同构思想）。缩放 $\alpha/r$ 控制增量的有效学习率。

参数量：单层 $2 \times in \times r$ vs 全量 $in \times out$。$d = 256$、$r = 8$：比例 $\frac{2 \times 256 \times 8}{256 \times 256} = 6.25\%$；只挂注意力投影（$4d^2$/层）时全局可训练参数占比更低。

<iframe src="resources/figures/lora-structure.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="LoRA：冻结主干 + 低秩旁路"></iframe>

<p align="center"><b>图 38.1</b>：W₀ 冻结, B·A 可训练</p>

### 1.2 机制分析

微调所需权重扰动 $\Delta W$ 的「内在秩」低的经验解释：任务适配（如指令格式）改变的是行为子空间而非全部表征——$\Delta W$ 集中在少数方向。数学上：任意 $\Delta W$ 逼近误差由第 $r+1$ 个奇异值之后能量决定（Eckart-Young），LoRA 赌的是谱衰减快。

### 1.3 训练效率结构

反向传播：$W_0$ 冻结则其参数梯度和 optimizer 状态（Adam 一二阶矩，2 倍参数量的内存）都不需要——**优化器状态的节省比参数本身更大**。梯度仍需流过 $W_0 x$ 路径（chain rule 经 $B A$ 到输入）。

<iframe src="resources/figures/param-vs-quality.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="低秩的饱和点"></iframe>

<p align="center"><b>图 38.2</b>：准确率 vs 可训练参数占比</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `LoRALinear` — 冻结 base
- `apply_lora` — 替换 q/k/v/o 投影
- `merge_lora` — W += BA 合并（推理零开销）

- 接口骨架：[`../../code/lab38-lora/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab38-lora/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab38-lora/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab38-lora/selftest.py)
- 实验入口：`cd code/lab38-lora && uv sync && uv run run_lab38.py`


