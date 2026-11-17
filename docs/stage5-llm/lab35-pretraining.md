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

### 1.2 学习率调度与梯度累积

Warmup + cosine（Lab 07 §1.4）。梯度累积：分 micro-batch 前向反传，梯度累加后一次更新。数学等价性：损失为各 micro-batch 均值的加权平均（权重 = 样本数比），梯度同。等效 batch = micro × accum，绕开内存限制。

### 1.3 规模的三方约束

固定算力下：$d$（宽度）↑ → 容量↑ 但易过拟合（数据不足时）；block_size ↑ → 长程依赖可学但有效样本数↓（同量文本切出的窗口变少）；数据量决定前两者的上限。TinyStories 的短文本特性使 block_size 128 已接近饱和——这本身是一个实验发现。

### 1.4 检查点与评估协议

固定 step 间隔存档（model + optimizer 状态 + step）；评估用持出集的 PPL 与生成样本双指标（PPL 度量拟合、样本暴露退化/重复等 PPL 看不出的病）。

## 2. 实现任务

```python
def pretrain(model, loader, cfg): ...      # warmup+cosine、梯度累积、checkpoint
def evaluate_ppl(model, val_loader): ...
def generate_samples(model, prompts, cfg): ...
```

配置基线：$d=256$、$L=4$、heads 4、block 128、$|V|=4k$（Lab 32）、AdamW lr $3\times10^{-4}$、warmup 200、cosine 到 5k-20k 步。

自测：

```python
# T1: 初始 val loss ≈ ln(4000) ≈ 8.29（±0.1）
# T2: 梯度累积等价性：accum=2×8 与 batch=16 一步更新后参数差 < 1e-6（同数据）
# T3: PPL = exp(loss) 数值核对
# T4: checkpoint 恢复：加载后继续训练的 loss 与不中断连续训练一致（同种子）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 训练进程（step 快照） | — | train/val PPL 双曲线、样本质量演化 |
| E2 | $d \in \{128, 256, 512\}$ | 数据/步数同 | val PPL、过拟合点 |
| E3 | block $\{64, 128, 256\}$ | — | val PPL、长生成连贯性 |
| E4 | lr 调度：恒定 vs cosine | — | 最终 PPL（连 Lab 07） |

## 4. 预期图表

- 图 1：train/val loss 双曲线 + 按快照的生成样本（早期乱码 → 词法正确 → 短句通顺）。
- 图 2：E2 三模型 val PPL：512 在小数据上后期反超差（过拟合）或持平。
- 图 3：E3 的 PPL 对比 + 256 block 的长文本生成更连贯。
- 图 4：E4 双曲线：cosine 后期下降更稳。

## 5. 观测指标

`val_ppl_curve`、`best_val_ppl`、`samples_by_step`、`ppl_by_config`。

## 6. 结果分析

1. 你的 val PPL 降到多少？PPL $= 15$ 意味着什么（用 1.1 解释）？
2. E2 中 512 维模型何时开始过拟合？与数据量（token 数）的关系？
3. 生成样本中出现的「重复循环」是什么阶段的典型病？（自回归分布的局部陷阱；与 greedy 解码相关，连 Lab 36。）

## 7. 验收清单

- [ ] T1-T4 通过；完整训练循环（调度、累积、checkpoint、评估）独立实现
- [ ] val PPL 明显低于 $e^{\ln V}$ 基线并趋于平稳（TinyStories 小模型典型 $5\sim10$）
- [ ] 生成样本达到「语法通顺的短故事」水平
