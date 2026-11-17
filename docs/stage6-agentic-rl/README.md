---
title: "第六阶段：Agentic RL"
stage: 6
status: 讲义已完成，实验代码骨架待建
---

核心问题：如何让语言模型通过与外部环境交互完成多步骤任务，并通过强化学习优化其行为？

这一阶段不再使用 TinyStories 作为主要任务数据，而是构建一个可验证的工具环境。它是全课程的最终汇聚：概率建模（轨迹分布）、策略梯度（第四阶段）、语言模型训练（第五阶段）、奖励设计（本阶段）在这里合为一体。

## 实验列表

| 实验 | 主题 | 知识块 |
|---|---|---|
| 实验 42 | 实现基础 Agent | Agent Loop、Tool Calling、Observation/Action、部分可观测 |
| 实验 43 | Agent Trajectory | 轨迹记录、工具调用统计、无效动作 |
| 实验 44 | Reward Design | Outcome/Process/Cost Reward、Reward Hacking |
| 实验 45 | Agent PPO / GRPO | Rollout、优势估计、KL 正则、On-policy 数据收集 |

四个知识块：

1. **Agent 与环境交互**：Agent Loop、Observation/Action、Tool Calling（Calculator、Structured Data Query、受限 Python Executor）、Environment State、Trajectory、Multi-step Interaction、Partially Observable Environment。
2. **轨迹与奖励**：Trajectory Distribution、Outcome Reward、Process Reward、Verifiable Reward、Sparse Reward、Reward Hacking、Task Evaluation。
3. **Agent 策略优化**：Agent Rollout、Policy Gradient、Advantage Estimation、PPO、GRPO、KL Regularization、On-policy Data Collection。
4. **综合项目**。

实验 45 的完整闭环：

$$
\text{Prompt Sampling} \rightarrow \text{Agent Rollout} \rightarrow \text{Trajectory Collection} \rightarrow \text{Reward Computation} \rightarrow \text{Advantage Estimation} \rightarrow \text{Policy Update} \rightarrow \text{Evaluation}
$$

## 概念接口

**对本阶段的前置依赖（全课程汇合点）**：

- 轨迹分布与回报期望 = 第一阶段的期望算子与 Monte Carlo 估计。
- 多步交互的信用分配 = 第四阶段的 Advantage/GAE。
- Policy = 第五阶段的语言模型；优化目标 = 实验 41 GRPO 迁移到工具交互环境。
- 奖励可验证性 = Lab 03 的「解析解对照」思想：可自动判分的任务环境。

**最终项目不是简单拼接**：Agentic RL 真正需要的是概率建模、策略梯度、轨迹采样、奖励设计和语言模型训练的结合。VAE、GAN 等生成模型建立的是生成建模能力，并不意味着最终 Agent 必须使用这些模型。

## 工程要求

Agent Loop 运行时、Trajectory 序列化存储、Reward 计算管线、Rollout Worker（批量并行采样与失败隔离）。

## 阶段验收

实验 45 中比较训练前后的任务成功率、工具调用效率、无效动作比例和 KL Divergence，能对训练过程做出定量分析。

## 综合项目：Agentic RL

构建一个具有工具调用能力的 Agent，在一个有限、可验证的任务环境中完成多步骤任务。要求：

- 有明确的环境定义。
- 有明确的 Action Space。
- 有可验证的 Reward。
- 能记录完整 Trajectory。
- 能完成策略训练。
- 能独立评估模型效果。
- 能对训练过程进行分析。

## 选修内容

Model-based RL、World Model、Planning、Multi-Agent RL。这些内容不属于当前主线的必要前置，主线完成后再按需补充。
