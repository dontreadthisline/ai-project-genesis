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

### 1.2 Causal mask

注意力 logits 加上三角掩码 $M_{ij} = -\infty\ (j > i)$（Lab 14 §1.1）后，位置 $t$ 的表示严格不含 $x_{>t}$ 信息。**掩码是唯一需要的位置级防泄漏机制**：配合 shift，整条序列的 T 个位置同时训练而无泄漏。验证方法：打乱 target 的未来部分，loss 不应变化（信息论上的独立性检查）。

### 1.3 Padding 与 loss mask

批量打包变长序列：右侧 pad。pad 位置的 loss 权重置 0（label = -100 的 ignore_index 惯例），否则模型学习「预测 pad」污染分布。损失：

$$
\mathcal L = \frac{\sum_t \mathbb 1[\text{label}_t \neq \text{pad}]\, \text{CE}_t}{\sum_t \mathbb 1[\text{label}_t \neq \text{pad}]}
$$

### 1.4 Sequence packing

短样本拼接到 block_size（如 512），一条样本承载多个文档。**必须加文档边界掩码**（block 内第 k 个文档的位置不 attend 跨文档），否则模型学到「上个文档的结尾预测下个文档的开头」——分布错误。工程收益：pad 浪费从 30%+ 降到近 0。

## 2. 实现任务

```python
def make_shift_pairs(ids, block_size): ...    # 滑窗切 -> (input, target)
def collate(batch, pad_id): ...               # pad + labels(-100)
def pack_sequences(docs, block_size): ...     # 拼接 + doc_mask
def masked_ce_loss(logits, labels): ...       # 1.3 公式
```

自测：

```python
# T1: shift 正确性：target[i] == input[i+1]（逐位）
# T2: pad 不计损失：全 pad 批的 loss == 0（0/0 以 0 处理）
# T3: 掩码独立性：把 target 的后半替换为随机 token，前半位置的 loss 不变
#     （位置 < k 的预测不依赖 target > k —— 验证无泄漏）
# T4: packing 计数：N 个文档 pack 后总 token 利用率 > 95%
# T5: 文档边界：pack 后第一个文档的 attention 最后一行不含第二文档 token
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 有无 causal mask（小模型几步训练） | — | loss 差异（无掩码时 loss 异常低 = 泄漏） |
| E2 | pad 计损 vs 不计损 | 变长批 | 训练动态、生成是否吐 pad |
| E3 | packing vs 不打包 | 同数据 | 有效 token 比例、吞吐 |
| E4 | block_size $\{64, 256\}$ | — | 样本数与上下文依赖的权衡 |

## 4. 预期图表

- 图 1：E1 双 loss 曲线：泄漏版迅速降到不可能的低值（把「答案」移进上下文）。
- 图 2：padding 浪费率 vs 批内长度分布（直方图叠加）。
- 图 3：E3 的吞吐对比柱状图。
- 图 4：一个 packed block 的可视化（文档边界、掩码热图）。

## 5. 观测指标

`effective_token_ratio`、`leak_test_delta`（应≈0）、`pad_waste`。

## 6. 结果分析

1. T3 的独立性测试为什么是「管线正确性」的最强测试？（它是信息论层面的验证，与实现细节无关。）
2. E4 中小 block 丢失了什么学习能力？对 TinyStories 这种短文本影响多大？
3. 计算：block=256、$d=256$ 时一条样本的前向 FLOPs（Lab 34 的参数量公式将复用）。

## 7. 验收清单

- [ ] T1-T5 通过；shift/pad/pack/mask 四件套全自实现
- [ ] 能用因果分解公式解释 shift 的必然性
- [ ] 理解泄漏测试与文档边界掩码这两个最常见 bug 源
