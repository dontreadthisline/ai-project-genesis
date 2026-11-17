---
title: "第四阶段：强化学习"
stage: 4
status: 讲义已完成，实验代码骨架待建
---

核心问题：当模型不再从固定数据集中学习，而是通过与环境交互获得反馈时，应该如何优化策略？

主要环境：Gymnasium。从表格方法到深度方法、从值方法到策略方法、从离散控制到连续控制，共三条递进线。

## 实验列表

| 实验 | 主题 | 知识块 |
|---|---|---|
| 实验 23 | FrozenLake 环境建模 | MDP、状态/动作/奖励、转移概率 |
| 实验 24 | 动态规划 | Bellman 方程、Policy/Value Iteration |
| 实验 25 | SARSA 与 Q-Learning | MC、TD、On/Off-policy、Bootstrapping |
| 实验 26 | 实现 DQN | 函数近似、Experience Replay、Target Network |
| 实验 27 | DQN 消融实验 | Double DQN、Dueling DQN、Q 值过高估计 |
| 实验 28 | REINFORCE | 策略梯度定理、Baseline、方差缩减 |
| 实验 29 | Actor-Critic | 优势函数、价值估计引导策略 |
| 实验 30 | 实现 PPO | 重要性采样、Clipped Objective、GAE |
| 实验 31 | Pendulum 连续控制 | DDPG、TD3、最大熵 RL（SAC） |

六个知识块：

1. **MDP 与动态规划**：Markov Decision Process、Return、Discount Factor、State/Action Value、Bellman 方程与最优性方程、Policy Iteration、Value Iteration。
2. **Model-free RL**：Monte Carlo、Temporal Difference、SARSA、Q-Learning、On/Off-policy、Bootstrapping。
3. **Deep Q-Network**：Function Approximation、Experience Replay、Target Network、Double/Dueling DQN。
4. **策略梯度与 Actor-Critic**：Stochastic Policy、Policy Gradient Theorem、REINFORCE、Baseline、Advantage Function、GAE。
5. **PPO**：Importance Sampling、Probability Ratio、Clipped Objective、Entropy Regularization、KL Constraint。
6. **连续控制与 SAC**：Continuous Action Space、Deterministic Policy Gradient、DDPG、TD3、Maximum Entropy RL、Entropy Temperature。

## 概念接口

**对本阶段的前置依赖**：

- 实验 28 的策略梯度 $\nabla J(\theta) = \mathbb E[\nabla_\theta \log \pi_\theta \cdot G]$ 是 Lab 03 Monte Carlo 期望估计的实例：单条轨迹回报就是一次采样，Baseline（实验 28）与 GAE（实验 30）都是 Lab 03「方差缩减」主题的延续。
- 实验 24 的 Bellman 期望 $v(s) = \mathbb E[r + \gamma v(s')]$ 是 Lab 02 条件期望的递归形态。
- 实验 26/29 的网络与优化器来自第二阶段；实验 30 的 KL 约束是 Lab 04 的直接应用。

**后续阶段的复用**：

- 实验 30 的 PPO 目标（ratio、clip、KL）原样出现在第五阶段 GRPO（实验 41）与第六阶段 Agent PPO/GRPO（实验 45），只是「动作」从 Gymnasium 动作变成 token 序列。
- 实验 26 的 Replay Buffer 与训练稳定性经验（target network 消融）是理解 off-policy 语言模型训练的参照系。

## 工程要求

Gymnasium 环境管理、Rollout 收集、Replay Buffer、训练与评估循环、指标可视化（episode return、策略熵、KL）。

## 阶段验收

能够独立实现：Tabular Q-Learning、DQN（含消融）、REINFORCE、PPO、SAC。

## 阶段项目

Gymnasium RL 算法库：统一环境交互、轨迹缓冲、训练、评估和可视化接口，表格算法与深度算法共用同一套评估协议（固定种子、多 run 均值与方差带）。
