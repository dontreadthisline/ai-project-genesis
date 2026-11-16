---
title: "第五阶段：LLM 预训练与后训练"
stage: 5
status: 讲义撰写中
---

核心问题：如何将前面学习的深度学习和强化学习机制应用到自回归语言模型？

主要数据集：TinyStories。本阶段是全课程的汇聚点：训练损失来自第一阶段，网络结构来自第二阶段，策略优化来自第四阶段。

## 实验列表

| 实验 | 主题 | 知识块 |
|---|---|---|
| 实验 32 | Tokenizer 实验 | BPE、词表大小与序列长度 |
| 实验 33 | 语言模型数据管线 | Causal Mask、Input/Target Shift、Padding、Packing |
| 实验 34 | 实现简化 GPT | Embedding、位置编码、Multi-Head Attention、LM Head |
| 实验 35 | TinyStories GPT 预训练 | NTP、Teacher Forcing、Perplexity、梯度累积 |
| 实验 36 | 推理优化 | Greedy/Temperature/Top-k/Top-p、KV Cache、Prefill/Decode |
| 实验 37 | SFT | 指令数据集、Prompt/Completion Masking |
| 实验 38 | LoRA 对照 | 参数高效微调、可训练参数量与速度 |
| 实验 39 | DPO | 偏好数据、Bradley-Terry、参考模型对数比 |
| 实验 40 | Reward Model | 偏好数据到奖励估计 |
| 实验 41 | 简化 GRPO | 组采样、相对优势、奖励归一化、KL 正则 |

七个知识块：

1. **Tokenizer 与数据管线**：Character/Word/Subword、Vocabulary、BPE、Token IDs、Sequence Construction、Causal Language Modeling。
2. **Transformer 与 GPT**：Embedding、Positional Encoding、Causal Mask、FFN、Transformer Block、LayerNorm、LM Head。
3. **预训练**：Next Token Prediction、Cross Entropy、Teacher Forcing、Validation Loss、Perplexity、LR Schedule、Gradient Accumulation、Checkpoint。
4. **推理**：Autoregressive Decoding、Greedy、Temperature、Top-k/Top-p、KV Cache、Prefill/Decode 两阶段。
5. **SFT 与 LoRA**：Instruction Dataset、Masking、LoRA、Parameter-Efficient Fine-Tuning。
6. **偏好优化**：Preference Dataset、Reward Model、Bradley-Terry、DPO、RLHF、PPO-based RLHF。
7. **GRPO**：Group Sampling、Relative Advantage、Reward Normalization、Clipped Policy Objective、KL Regularization、Token-level Policy Optimization。

## 概念接口

**对本阶段的前置依赖**：

- 实验 35 的训练目标 = Lab 05 的结论：最小化 NLL = 最小化与经验分布的交叉熵；Perplexity 是该交叉熵的指数。
- 实验 34 的网络 = 实验 14 Transformer 加因果掩码与 LM Head。
- 实验 39 的 DPO 隐式奖励 = policy 与 reference 的对数比，其期望控制项是 Lab 04 反向 KL 的变体；实验 41 的 GRPO 目标 = 实验 30 PPO 目标加组内相对优势，组内回报均值与标准差的归一化是 Lab 03 估计量标准化的翻版。

**后续阶段的复用**：

- 实验 34-41 的全部组件在第六阶段组装：policy = 实验 35/37 的语言模型，rollout = 工具交互轨迹，reward = 可验证奖励，优化器 = 实验 41 的 GRPO。

## 工程要求

数据管线（分词、打包、掩码）、序列处理、Checkpoint 管理、推理优化（KV Cache 的内存与速度测量）。

## 阶段验收

能够独立完成一个小型语言模型的：数据准备、预训练、自回归推理、SFT、LoRA、DPO、简化 PPO/GRPO 实验。

## 阶段项目

小型 LLM 训练与后训练系统：从原始 TinyStories 文本到可对话微调模型的完整链路，每个环节可独立评估（loss/perplexity、生成质量、指令遵循率、偏好准确率、任务准确率）。
