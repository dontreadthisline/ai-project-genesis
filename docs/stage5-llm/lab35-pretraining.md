---
title: "Lab 35：TinyStories GPT 预训练"
lab: 35
stage: 5
---

对应大纲实验 35。依赖 Lab 33/34。全课程第一个「完整训练一个模型」的实验。

## 0. 理论问题

1. 训练 loss 与验证 loss 的分化何时出现？说明什么？
2. Perplexity 与 loss 的关系？它直观上度量什么？
3. 模型规模、上下文长度、数据量三者如何互相约束？

## 1. 数学基础

### 1.1 训练目标与困惑度

$$
\mathcal L = -\frac1T \sum_t \log p_\theta(x_t | x_{<t}), \qquad
\mathrm{PPL} = e^{\mathcal L}
$$

PPL = 「模型在每个位置上等效的均匀选择数」：PPL $= 10$ 意味着与「从 10 个等概率候选中选对」同水平的不确定性（Lab 04 的 $H(P, Q)$ 视角：$e^{H(P,Q)}$ 是交叉熵的指数）。

<iframe src="resources/figures/train-val-divergence.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="过拟合的分歧点"></iframe>

<p align="center"><b>图 35.1</b>：train/val 双曲线</p>

### 1.2 学习率调度与梯度累积

Warmup + cosine（Lab 07 §1.4）。梯度累积：分 micro-batch 前向反传，梯度累加后一次更新。数学等价性：损失为各 micro-batch 均值的加权平均（权重 = 样本数比），梯度同。等效 batch = micro × accum，绕开内存限制。

### 1.3 规模的三方约束

固定算力下：$d$（宽度）↑ → 容量↑ 但易过拟合（数据不足时）；block_size ↑ → 长程依赖可学但有效样本数↓（同量文本切出的窗口变少）；数据量决定前两者的上限。TinyStories 的短文本特性使 block_size 128 已接近饱和——这本身是一个实验发现。

<iframe src="resources/figures/ppl-scale.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="困惑度与规模"></iframe>

<p align="center"><b>图 35.2</b>：PPL vs d</p>

### 1.4 检查点与评估协议

固定 step 间隔存档（model + optimizer 状态 + step）；评估用持出集的 PPL 与生成样本双指标（PPL 度量拟合、样本暴露退化/重复等 PPL 看不出的病）。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `pretrain` — warmup+cosine、梯度累积、checkpoint
- `evaluate_ppl` — evaluate_ppl
- `generate_samples` — generate_samples

- 接口骨架：[`../../code/lab35-pretraining/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab35-pretraining/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab35-pretraining/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab35-pretraining/selftest.py)
- 实验入口：`cd code/lab35-pretraining && uv sync && uv run run_lab35.py`


