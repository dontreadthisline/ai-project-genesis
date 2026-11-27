---
title: "Lab 37：SFT"
lab: 37
stage: 5
---

对应大纲实验 37。依赖 Lab 35 的基座模型。

## 0. 理论问题

1. SFT 与预训练在数学上是同一个目标，差别只在数据与掩码——差在哪？
2. Prompt loss 掩码为什么重要？
3. 小指令集上全量微调的过拟合形态？

## 1. 数学基础

### 1.1 目标的形式不变与数据的形式改变

SFT 仍是因果语言建模（最小化 NLL，Lab 05），但数据从「无标注文本」变为「指令-回复对」。条件视角：预训练学 $p(x)$（文档分布），SFT 集中学习 $p(y \mid x)$（给定指令的回复分布）——同一模型、同一损失、不同的条件化数据配比。

### 1.2 Prompt masking

序列 $[\text{prompt}]\,[\text{completion}]$，label 只在 completion 部分：

$$
\mathcal L = -\frac{1}{|y|}\sum_{t \in \text{completion}} \log p_\theta(y_t \mid \text{prompt}, y_{<t})
$$

不掩 prompt 的后果：模型同时学「生成指令」（把用户输入也当生成目标），有限的指令分布污染生成分布——尤其指令模板高度重复时，模型过拟合模板而非映射。

<iframe src="resources/figures/prompt-mask.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="SFT：只对回复计损"></iframe>

<p align="center"><b>图 37.1</b>：序列的两段与掩码</p>

### 1.3 与 RLHF 的位置关系

SFT 是对齐流水线第一级（SFT → RM → RLHF/DPO，本阶段 37 → 40 → 39/41）。SFT 用行为克隆给出「格式的先验」，偏好优化再调「质量的方向」——没有 SFT 的格式先验，偏好数据都难以构造。

<iframe src="resources/figures/forgetting.html" style="width:100%;height:354px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="灾难性遗忘的量级"></iframe>

<p align="center"><b>图 37.2</b>：SFT 前后的两个指标</p>

## 2. 实现任务

构造小指令集（自动生成、可验证）：算术问答（"3+4=?"）、格式转换（大小写、倒序）、单步推理（比大小）。约 2k 对，模板多样化（5 种问法）。

实现清单（签名与 TODO 见骨架源码）：

- `build_sft_dataset` — (prompt_ids, completion_ids)
- `sft_collate` — labels: prompt 部分全 -100
- `sft_train` — 与 pretrain 共用循环，换 loader

- 接口骨架：[`../../code/lab37-sft/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab37-sft/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab37-sft/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab37-sft/selftest.py)
- 实验入口：`cd code/lab37-sft && uv sync && uv run run_lab37.py`


