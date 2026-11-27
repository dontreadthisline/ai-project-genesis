---
title: "Lab 34：实现简化 GPT"
lab: 34
stage: 5
---

对应大纲实验 34。依赖 Lab 14（Transformer Block）、Lab 32/33。

## 0. 理论问题

1. GPT 相比 Lab 14 的 Transformer 增加了哪些组件？各自职责？
2. 参数量如何按层分解？$12Ld^2$ 从哪来？
3. 因果掩码下的「一次前向 = T 个训练样本」如何与生成模式统一？

## 1. 数学基础

### 1.1 架构清单

$$
h_0 = W_e\, x + W_p\, p, \qquad
h_l = h_{l-1} + \mathrm{Block}(\mathrm{LN}(h_{l-1})), \quad l = 1..L
$$
$$
h = \mathrm{LN}_f(h_L), \qquad \text{logits} = h\, W_e^T \ \ (\text{权重共享})
$$

Block 内：causal MHA + FFN（Lab 14 §1.4 pre-norm）。组件对照 Lab 14 的增量：可学习位置嵌入 $W_p$、最终 LN、LM Head 与 embedding 权重共享（参数节省 $|V| d$，且输入输出空间对齐）。

### 1.2 参数量（推导）

每个 Block：MHA 四个投影 $4d^2$ + FFN $2 \times d \times 4d = 8d^2$，共 $12d^2$（LN 参数可忽略）。总计：

$$
P \approx L \cdot 12d^2 + |V|\, d \ (\text{embedding，共享时不重复计})
$$

$d = 256$、$L = 4$、$|V| = 4k$：$12 \cdot 4 \cdot 65536 + 1M \approx 4.2M$。

### 1.3 训练与生成的统一

训练：一次前向出全部 T 个位置的 logits，与 shift target 算交叉熵（Lab 33）。生成：迭代——forward 取最后一个位置的 logits，采样，append。同一网络两种用法，区别只在「用全部位置还是最后一个位置」。KV cache（Lab 36）将把生成模式的重复前向优化掉。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `GPT` — (B, T) -> logits (B, T, V)

- 接口骨架：[`../../code/lab34-gpt/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab34-gpt/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab34-gpt/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab34-gpt/selftest.py)
- 实验入口：`cd code/lab34-gpt && uv sync && uv run run_lab34.py`


