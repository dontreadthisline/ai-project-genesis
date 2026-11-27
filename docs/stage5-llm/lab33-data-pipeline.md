---
title: "Lab 33：语言模型数据管线"
lab: 33
stage: 5
---

对应大纲实验 33。依赖 Lab 32。本实验产物是 Lab 35 预训练的直接输入。

## 0. 理论问题

1. 自回归训练的 input/target 对如何构造？shift 一步的数学含义？
2. Causal mask 如何让一个序列同时提供 T 个训练样本？
3. Loss masking 与 packing 各解决什么问题？

## 1. 数学基础

### 1.1 因果分解与 teacher forcing

$$
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

训练即最大化对数似然（= 最小化 NLL = 交叉熵，Lab 05 §1.5）。Teacher forcing：输入真实前缀 $x_{<t}$ 预测 $x_t$（不是自己的生成）。批实现：一个序列同时是 T 个样本——位置 $t$ 的输出头只用 $\le t$ 的输入。

$$
\text{input} = (x_1, \dots, x_T), \qquad \text{target} = (x_2, \dots, x_{T+1})
$$

target 是 input 左移一位（shift）。一步前向 + 一次交叉熵（按位置平均）完成 T 个预测的损失。

<iframe src="resources/figures/shift-mask.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="自回归的输入-目标对"></iframe>

<p align="center"><b>图 33.1</b>：input/target 错位与掩码</p>

### 1.2 Causal mask

注意力 logits 加上三角掩码 $M_{ij} = -\infty\ (j > i)$（Lab 14 §1.1）后，位置 $t$ 的表示严格不含 $x_{>t}$ 信息。**掩码是唯一需要的位置级防泄漏机制**：配合 shift，整条序列的 T 个位置同时训练而无泄漏。验证方法：打乱 target 的未来部分，loss 不应变化（信息论上的独立性检查）。

### 1.3 Padding 与 loss mask

批量打包变长序列：右侧 pad。pad 位置的 loss 权重置 0（label = -100 的 ignore_index 惯例），否则模型学习「预测 pad」污染分布。损失：

$$
\mathcal L = \frac{\sum_t \mathbb 1[\text{label}_t \neq \text{pad}]\, \text{CE}_t}{\sum_t \mathbb 1[\text{label}_t \neq \text{pad}]}
$$

### 1.4 Sequence packing

短样本拼接到 block_size（如 512），一条样本承载多个文档。**必须加文档边界掩码**（block 内第 k 个文档的位置不 attend 跨文档），否则模型学到「上个文档的结尾预测下个文档的开头」——分布错误。工程收益：pad 浪费从 30%+ 降到近 0。

<iframe src="resources/figures/packing-boundary.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="Sequence Packing 的边界"></iframe>

<p align="center"><b>图 33.2</b>：同 block 两个文档 + 边界掩码</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `make_shift_pairs` — 滑窗切 -> (input, target)
- `collate` — pad + labels(-100)
- `pack_sequences` — 拼接 + doc_mask
- `masked_ce_loss` — 1.3 公式

- 接口骨架：[`../../code/lab33-data-pipeline/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab33-data-pipeline/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab33-data-pipeline/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab33-data-pipeline/selftest.py)
- 实验入口：`cd code/lab33-data-pipeline && uv sync && uv run run_lab33.py`


