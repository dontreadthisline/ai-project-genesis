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

### 1.2 机制分析

微调所需权重扰动 $\Delta W$ 的「内在秩」低的经验解释：任务适配（如指令格式）改变的是行为子空间而非全部表征——$\Delta W$ 集中在少数方向。数学上：任意 $\Delta W$ 逼近误差由第 $r+1$ 个奇异值之后能量决定（Eckart-Young），LoRA 赌的是谱衰减快。

### 1.3 训练效率结构

反向传播：$W_0$ 冻结则其参数梯度和 optimizer 状态（Adam 一二阶矩，2 倍参数量的内存）都不需要——**优化器状态的节省比参数本身更大**。梯度仍需流过 $W_0 x$ 路径（chain rule 经 $B A$ 到输入）。

## 2. 实现任务

```python
class LoRALinear:
    def __init__(self, base_linear, r, alpha): ...   # 冻结 base
    def forward(self, x): ...                        # base(x) + scale * B(A(x))
def apply_lora(model, targets, r, alpha): ...        # 替换 q/k/v/o 投影
def merge_lora(model): ...                           # W += BA 合并（推理零开销）
```

自测：

```python
# T1: 零初始化：B=0 时 LoRALinear 输出 == base 输出（逐位）
# T2: 参数量公式：2*d*r*层数*投影数 vs 实际可训练参数（误差 <1%）
# T3: 合并等价：merge 后的前向 == 未合并前向（1e-5）
# T4: 冻结检查：base 参数梯度恒为 0
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | Full SFT vs LoRA $r \in \{2, 8, 32\}$ | 同任务同步数 | 任务准确率、可训练参数量 |
| E2 | LoRA 挂载位置：attn only / FFN only / 全部 | — | 准确率 |
| E3 | 训练吞吐与内存：full vs LoRA | — | tokens/s、峰值内存 |
| E4 | checkpoint 大小与切换 | — | 保存体积、多任务切换开销 |

## 4. 预期图表

- 图 1：准确率 vs 可训练参数量（对数 x 轴）：r=8 已接近 full 的 95%+ 效果。
- 图 2：E2 柱状图：attn 投影挂载收益最大（行为适配集中在注意力）。
- 图 3：E3 内存柱状：优化器状态的节省主导（full 的 ~3 倍参数内存 vs LoRA）。
- 图 4：E4 的多任务 LoRA 池示意（一个基座 + N 个 6% 适配器）。

## 5. 观测指标

`acc_vs_trainable_params`、`memory_peak`、`ckpt_size`、`throughput`。

## 6. 结果分析

1. E1 中 r 的饱和点在哪？与任务复杂度（三种指令任务的差异）有什么关系？
2. E3 中吞吐 LoRA 略低于 full 的原因？（额外的 A/B 前向。）
3. LoRA 与 Lab 11 残差在「低秩修正 + 零初始化」上的同构性，说明这种「冻结主干 + 小容量修正」设计的普适性——再举一个课程中出现过的例子（Lab 38 之外）。

## 7. 验收清单

- [ ] T1-T4 通过；LoRA 前向/合并/冻结语义正确
- [ ] 完成 full vs LoRA 的准确率-参数-内存三维对照
- [ ] r=8 达到 full SFT 准确率的 95%（构造任务上）
