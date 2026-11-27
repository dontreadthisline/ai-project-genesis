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

### 1.3 与 RLHF 的位置关系

SFT 是对齐流水线第一级（SFT → RM → RLHF/DPO，本阶段 37 → 40 → 39/41）。SFT 用行为克隆给出「格式的先验」，偏好优化再调「质量的方向」——没有 SFT 的格式先验，偏好数据都难以构造。

## 2. 实现任务

构造小指令集（自动生成、可验证）：算术问答（"3+4=?"）、格式转换（大小写、倒序）、单步推理（比大小）。约 2k 对，模板多样化（5 种问法）。

实现清单（签名与 TODO 见骨架源码）：

- `build_sft_dataset` — (prompt_ids, completion_ids)
- `sft_collate` — labels: prompt 部分全 -100
- `sft_train` — 与 pretrain 共用循环，换 loader

- 接口骨架：[`../../code/lab37-sft/skeleton.py`](../../code/lab37-sft/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab37-sft/selftest.py`](../../code/lab37-sft/selftest.py)
- 实验入口：`cd code/lab37-sft && uv sync && uv run run_lab37.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 掩码：全序列损失 vs 仅 completion | — | 指令遵循率、回复模板化程度 |
| E2 | 训练 epoch $\{1, 3, 10\}$ | — | 训练/验证准确率分化 |
| E3 | SFT 前 vs 后 | — | 指令遵循率、通用 PPL（遗忘检查） |
| E4 | 指令多样性 $\{1 种模板, 5 种\}$ | — | 模板鲁棒性 |

## 4. 预期图表

- 图 1：E1 双曲线 + 生成样例对照：不掩码版本学会「自问自答」的复读行为。
- 图 2：E2 过拟合曲线：epoch 10 时 train acc 100%、验证 acc 回落。
- 图 3：E3 的任务准确率跳变（SFT 前后）+ 通用 PPL 轻微上升（灾难性遗忘的量级）。
- 图 4：E4 的模板迁移准确率。

## 5. 观测指标

`instruction_follow_rate`、`task_accuracy`、`generic_ppl_delta`、`mask_ablation`。

## 6. 结果分析

1. E3 中通用 PPL 上升多少？说明小数据 SFT 的遗忘是可接受的还是严重的？
2. E4 说明模板多样性的作用机制？（防止把「格式」当作任务特征。）
3. 若 completion 里混入错误答案（10% 噪声），SFT 学到什么？（分布拟合无所谓对错——引出 Lab 40/41 用奖励纠偏。）

## 7. 验收清单

- [ ] T1-T4 通过；prompt 掩码与损失归一化正确
- [ ] 指令遵循率从接近 0 提升到 80%+（构造任务上）
- [ ] 能叙述 SFT 在对齐流水线中的位置与局限
