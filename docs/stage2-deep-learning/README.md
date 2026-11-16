---
title: "第二阶段：深度学习计算基础与序列建模"
stage: 2
status: 讲义撰写中
---

核心问题：神经网络如何通过梯度优化学习复杂函数？不同网络结构为什么具有不同的表达能力和归纳偏置？

主要数据集：MNIST。本阶段产出的训练框架将在第三阶段（生成模型）被直接复用。

## 实验列表

| 实验 | 主题 | 知识块 |
|---|---|---|
| 实验 6 | 从零实现 MLP（784-256-128-10） | 计算图、反向传播、链式法则、VJP |
| 实验 7 | 优化器对照：SGD / Momentum / Adam / AdamW | 优化机制、学习率调度 |
| 实验 8 | Batch Size 对梯度估计的影响 | Mini-batch 优化、梯度偏差与方差 |
| 实验 9 | MLP 与 CNN 对照 | 卷积、局部连接、参数共享、平移等变性 |
| 实验 10 | 卷积结构的归纳偏置（像素位置打乱） | 感受野、Kernel Size、深度 |
| 实验 11 | Residual Connection 与深度 | 梯度传播、训练稳定性 |
| 实验 12 | 归一化与正则化对照 | BatchNorm、LayerNorm、Dropout、Weight Decay |
| 实验 13 | RNN 与 LSTM 的长期依赖问题 | BPTT、梯度消失/爆炸 |
| 实验 14 | Attention 与 Transformer | Self-Attention、字符级自回归预测 |

知识内容按四个块展开：

1. **神经网络与优化机制**：Tensor 与计算图、Forward/Backward、Chain Rule、Jacobian 与 VJP、SGD、Momentum、Adam/AdamW、Learning Rate Schedule、Mini-batch Optimization。
2. **CNN 与残差网络**：Convolution Operator、Kernel/Stride/Pooling、Local Connectivity、Parameter Sharing、Translation Equivariance、Receptive Field、Residual Connection。
3. **Normalization 与 Regularization**：BatchNorm、LayerNorm、Dropout、Weight Decay、Train/Eval Mode。
4. **Sequence Modeling**：RNN、BPTT、Vanishing/Exploding Gradient、LSTM/GRU、Self-Attention、Transformer。

## 概念接口

**对本阶段的前置依赖**：

- 实验 6 的分类损失就是 Lab 04/05 的结论实例化：训练即最小化 NLL = 最小化与经验分布的交叉熵 = KL 投影。
- 实验 8 的 mini-batch 梯度本质上是 Lab 03 的 Monte Carlo 估计：全量梯度是期望，batch 梯度是采样估计，误差随 batch size 的平方根衰减。

**后续阶段的复用**：

- 实验 14 的 Transformer Block 是第五阶段 GPT（实验 34/35）的直接前置，届时只新增 Embedding、位置编码、LM Head 与因果掩码。
- 本阶段的统一训练框架（模型切换、优化器切换、日志、验证、Checkpoint）是第三阶段所有生成模型的训练基座。
- 实验 12 的 LayerNorm 与 Train/Eval Mode 语义贯穿后续所有模型。

## 工程要求（MLX）

Module 与自动微分、Optimizer、训练循环、数据加载。Python 与 MLX 不单独开课，随实验逐步深入。

## 阶段验收

能够独立实现：MLP、CNN、基础残差网络、RNN/LSTM、简化 Transformer、SGD 与 Adam 等优化算法。

## 阶段项目

统一 MNIST 训练框架：支持模型切换、优化器切换、训练日志、验证和 Checkpoint，支持在固定接口下对比任意两个模型的训练曲线与测试表现。
