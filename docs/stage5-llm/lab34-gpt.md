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

- 接口骨架：[`../../code/lab34-gpt/skeleton.py`](../../code/lab34-gpt/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab34-gpt/selftest.py`](../../code/lab34-gpt/selftest.py)
- 实验入口：`cd code/lab34-gpt && uv sync && uv run run_lab34.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 权重共享 vs 独立 LM head | 参数对齐 | 收敛速度 |
| E2 | $d \in \{64, 128, 256\}$ | 层数固定 | 小数据 loss（容量-数据匹配） |
| E3 | 无位置嵌入 / 有 | — | 生成文本的顺序结构（连 Lab 14 E3） |
| E4 | 短训练（1k 步）后各层 attention 模式 | — | 头的分工初现 |

## 4. 预期图表

- 图 1：参数量分解饼图（MHA/FFN/emb 占比，FFN 占 $2/3$）。
- 图 2：E2 loss 曲线：小数据上大模型并不占优（过拟合）。
- 图 3：E3 生成对照：无位置信息时局部合理全局乱序（复现 Lab 14 结论）。
- 图 4：E4 注意力热图若干（前层局部、某头关注句首）。

## 5. 观测指标

`params_count`、`params_breakdown`、`sanity_loss`（初始 ≈ ln V）。

## 6. 结果分析

1. 初始 loss 应为 $\ln|V|$（随机初始化 = 均匀分布，Lab 04 的均匀熵）。你的实测值是多少？偏离说明什么（初始化尺度问题）？
2. E1 的共享收益在大词表下更明显还是更弱？
3. 用 1.2 公式估算：GPT-2 small（$L=12, d=768, V=50k$）参数量，与公开值 124M 对照。

## 7. 验收清单

- [ ] T1-T5 通过；能独立写出完整 GPT 前向
- [ ] 能推导 $12Ld^2$ 并做参数分解
- [ ] 因果性测试通过（T2 是架构正确性的关键测试）
