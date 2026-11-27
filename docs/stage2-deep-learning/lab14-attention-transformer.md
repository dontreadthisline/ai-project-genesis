---
title: "Lab 14：Attention 与 Transformer"
lab: 14
stage: 2
---

对应大纲实验 14。这是第五阶段 GPT（Lab 34/35）的直接前置。

## 0. 理论问题

1. Attention 作为「加权平均」在计算什么？权重从哪来？
2. 点积注意力为什么要除以 $\sqrt{d}$？
3. 多头、因果掩码、位置编码各自解决什么问题？

## 1. 数学基础

### 1.1 缩放点积注意力

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
$$

每个查询对全部键算相似度，softmax 归一化为分布，对值加权求和：一种内容寻址的软查找。$\sqrt{d_k}$ 缩放的原因：设 $q, k$ 各分量独立零均值方差 1，则 $q \cdot k$ 方差为 $d_k$；不缩放时大 $d_k$ 下点积幅度 $\sim\sqrt{d_k}$，softmax 进入饱和区，梯度趋于零。除以 $\sqrt{d_k}$ 把 logits 方差归一到 1。

因果掩码：$M_{ij} = 0\ (j \le i)$，$-\infty\ (j > i)$，softmax 后未来位置权重严格为 0——自回归分解 $p(x_1,\dots,x_T) = \prod_t p(x_t \mid x_{<t})$ 的实现。

### 1.2 多头注意力

$$
\mathrm{head}_i = \mathrm{Attn}(QW_i^Q,\ KW_i^K,\ VW_i^V), \qquad
\mathrm{MHA}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) W^O
$$

单头是秩 1 的软查找倾向（一个注意力分布只能强调一种模式）；多头允许在不同子空间并行关注不同关系（句法/位置/复制），代价是每头维度 $d/h$。

### 1.3 位置编码

注意力本身是置换等变的（把输入重排，输出同样重排），序列顺序信息必须外注入。正弦编码：

$$
PE_{(p, 2i)} = \sin(p / 10000^{2i/d}), \qquad PE_{(p, 2i+1)} = \cos(p / 10000^{2i/d})
$$

相对位置 $p - q$ 的函数可由 $PE_p, PE_q$ 的线性变换表出（旋转视角），使模型能泛化到不同位置。可学习绝对位置编码是等价替代（Lab 34 用它）。

### 1.4 Transformer Block（pre-norm）

$$
h' = h + \mathrm{MHA}(\mathrm{LN}(h)), \qquad
h'' = h' + \mathrm{FFN}(\mathrm{LN}(h')), \qquad \mathrm{FFN}(x) = W_2\, \mathrm{GELU}(W_1 x)
$$

残差主干 + LN 前置（Lab 11 + Lab 12 的组合）：pre-norm 的恒等路径不被归一化打断，深堆稳定（GPT 系沿用；post-norm 需 warmup 且深堆不稳）。

## 2. 实现任务

字符级自回归：语料用 tiny-shakespeare 或自制中文短语语料（确保可离线生成）。模型 $d=128$、$h=4$、$L=4$、块大小 128。

实现清单（签名与 TODO 见骨架源码）：

- `attention` — 1.1 公式，softmax 数值稳定
- `MultiHeadAttention` — MultiHeadAttention
- `Block` — 1.4 pre-norm
- `MiniTransformer` — emb + pos + L*Block + ln_f + head

- 接口骨架：[`../../code/lab14-attention-transformer/skeleton.py`](../../code/lab14-attention-transformer/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab14-attention-transformer/selftest.py`](../../code/lab14-attention-transformer/selftest.py)
- 实验入口：`cd code/lab14-attention-transformer && uv sync && uv run run_lab14.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | RNN/LSTM vs Transformer（参数量对齐） | 同语料同预算 | loss、生成质量 |
| E2 | 层数 $\{1, 2, 4, 8\}$ | — | loss 与有效上下文利用 |
| E3 | 无位置编码 / 正弦 / 可学习 | — | loss 与生成中的顺序结构 |
| E4 | 训练时上下文长度 16 vs 128 | — | 推理时外推长度行为的差异 |

## 4. 预期图表

- 图 1：两模型 loss 曲线：Transformer 收敛更快（并行训练 + 直接访问长程依赖）。
- 图 2：层数-loss：4 层后收益递减（小语料上限）。
- 图 3：E3 生成样本对比：无位置编码的生成在局部合理但全局乱序（字符 n-gram 统计正确但词序破坏）。
- 图 4：注意力可视化（任一层任一头）：可见对角线 + 若干头呈均匀/局部模式。

## 5. 观测指标

`final_val_loss`、`gen_samples`、`params_count`、`attn_entropy_by_head`。

## 6. 结果分析

1. 用 1.1 的方差论证解释缩放因子的必要性；实验观察不除 $\sqrt{d}$ 时训练有何变化（可加做）。
2. E1 中 Transformer 的优势来源是哪两点？各自对应 1.x 的哪个机制？
3. E4 中短上下文模型在长生成时退化为什么？（分布外位置编码；自回归误差累积。）

## 7. 验收清单

- [ ] T1-T4 通过；能白板写出缩放点积注意力与 pre-norm Block
- [ ] 能推导 $q\cdot k$ 的方差并解释缩放
- [ ] 能说明因果掩码与自回归分解的对应
- [ ] 生成样本通顺度显著优于随机基线（人工评估 + loss 曲线）
