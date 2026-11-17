---
title: "课程总纲"
---

## 路线总览

$$
\text{Probability}
\rightarrow
\text{Deep Learning}
\rightarrow
\text{Generative Models}
\rightarrow
\text{RL}
\rightarrow
\text{LLM}
\rightarrow
\text{Agentic RL}
$$

三条主要实验线：MNIST（深度学习/生成模型）、Gymnasium（强化学习）、TinyStories（LLM）。Python 与 MLX 不单独设阶段，贯穿所有项目。

## 一、最终学习目标

通过一系列从简单到复杂的实验，最终具备独立实现和训练 Agentic RL 系统的能力。课程结束时应当能够：

1. 理解深度学习、生成模型和强化学习中的核心数学机制。
2. 使用 MLX 独立实现典型神经网络和训练算法。
3. 实现 VAE、GAN、Diffusion、Flow Matching 等生成模型。
4. 使用 Gymnasium 实现 DQN、PPO、SAC 等强化学习算法。
5. 实现小型 GPT，完成预训练、SFT、DPO 和简化的 PPO/GRPO 实验。
6. 构建具有工具调用能力的 Agent 环境，并完成轨迹采集、奖励计算和策略优化。

## 二、实验组织原则

每个实验按以下结构组织：

- 理论问题：当前实验要研究什么问题？
- 数学基础：需要哪些定义、公式和推导？
- 实现任务：需要实现哪些模型或算法？
- 对照实验：改变什么变量，比较什么结果？
- 观测指标：需要记录哪些数据？
- 结果分析：实验结果说明了什么？是否符合理论预期？

每个阶段最终形成一个可以独立运行的项目，而不是大量互不关联的代码片段。正式授课时，每阶段采用「核心实验 + 消融实验 + 阶段项目」三级结构。

## 三、阶段划分

### 第一阶段：概率建模与信息论基础（3 周，优先级最高）

核心问题：如何使用数学语言描述随机现象、概率分布，以及不同概率分布之间的关系？

- 1.1 概率空间与随机变量：概率测度、PMF/PDF/CDF、联合/边缘/条件分布、独立性
  - 实验 1：一维与二维概率分布
  - 实验 2：联合分布与条件分布
- 1.2 期望与随机过程：期望、方差、协方差、全期望、大数定律、Monte Carlo
  - 实验 3：Monte Carlo 估计
- 1.3 信息论与概率估计：Entropy、Cross Entropy、KL、MLE、MAP、Brier Score
  - 实验 4：概率分布度量
  - 实验 5：最大似然估计

阶段项目：概率分布实验库（分布采样、密度计算、可视化、数值统计）。

讲义目录：[stage1-probability/](stage1-probability/README.md)

### 第二阶段：深度学习计算基础与序列建模（3 周）

核心问题：神经网络如何通过梯度优化学习复杂函数？不同网络结构为什么具有不同的表达能力和归纳偏置？主要数据集：MNIST。

- 2.1 神经网络与优化：计算图、反向传播、VJP、SGD/Momentum/AdamW、LR Schedule
  - 实验 6：从零实现 MLP
  - 实验 7：优化器对照
  - 实验 8：Batch Size 对梯度估计的影响
- 2.2 CNN 与残差网络：卷积、局部连接、参数共享、平移等变性、感受野、残差连接
  - 实验 9：MLP 与 CNN 对照
  - 实验 10：卷积结构的归纳偏置
  - 实验 11：Residual Connection
- 2.3 Normalization 与 Regularization：BatchNorm、LayerNorm、Dropout、Weight Decay
  - 实验 12：归一化与正则化对照
- 2.4 序列建模：RNN、BPTT、梯度消失/爆炸、LSTM/GRU、Attention、Transformer
  - 实验 13：RNN 与 LSTM 的长期依赖问题
  - 实验 14：Attention 与 Transformer

阶段项目：统一 MNIST 训练框架。

### 第三阶段：生成模型（5 周）

核心问题：如何让神经网络学习数据分布，并从学习到的分布中生成新样本？主要数据集：MNIST。

- 3.1 VAE：潜变量模型、变分推断、ELBO、重参数化
  - 实验 15：实现 VAE
  - 实验 16：Reparameterization 与梯度传播
- 3.2 GAN：极小极大博弈、JS 散度、Mode Collapse、训练不稳定
  - 实验 17：实现基础 GAN
- 3.3 Normalizing Flow：变量替换、可逆变换、Jacobian 行列式
  - 实验 18：二维 Normalizing Flow
- 3.4 Diffusion：前向马尔可夫过程、噪声调度、反向过程、噪声预测、Score、DDPM/DDIM、PF-ODE
  - 实验 19：实现 DDPM
- 3.5 Flow Matching：概率路径、向量场、CFM、Rectified Flow
  - 实验 20：二维 Flow Matching
  - 实验 21：MNIST Flow Matching
  - 实验 22：Diffusion 与 Flow Matching 对照

阶段项目：MNIST 生成模型实验库。

### 第四阶段：强化学习（5-6 周，优先级最高）

核心问题：当模型不再从固定数据集中学习，而是通过与环境交互获得反馈时，应该如何优化策略？主要环境：Gymnasium。

- 4.1 MDP 与动态规划：Bellman 方程、Policy/Value Iteration
  - 实验 23：FrozenLake 环境建模
  - 实验 24：动态规划
- 4.2 Model-free RL：MC、TD、SARSA、Q-Learning、On/Off-policy
  - 实验 25：SARSA 与 Q-Learning
- 4.3 DQN：函数近似、Experience Replay、Target Network、Double/Dueling
  - 实验 26：实现 DQN
  - 实验 27：DQN 消融实验
- 4.4 策略梯度与 Actor-Critic：Policy Gradient Theorem、REINFORCE、Baseline、GAE
  - 实验 28：REINFORCE
  - 实验 29：Actor-Critic
- 4.5 PPO：重要性采样、Clipped Objective、KL 约束
  - 实验 30：实现 PPO
- 4.6 连续控制与 SAC：DDPG、TD3、最大熵 RL
  - 实验 31：Pendulum 连续控制（DDPG → TD3 → SAC）

阶段项目：Gymnasium RL 算法库。

### 第五阶段：LLM 预训练与后训练（5-6 周，优先级最高）

核心问题：如何将深度学习和强化学习机制应用到自回归语言模型？主要数据集：TinyStories。

- 5.1 Tokenizer 与数据管线：BPE、Token IDs、Causal LM、Packing
  - 实验 32：Tokenizer 实验
  - 实验 33：语言模型数据管线
- 5.2 Transformer 与 GPT：Embedding、位置编码、MHA、Causal Mask
  - 实验 34：实现简化 GPT
- 5.3 预训练：NTP、Teacher Forcing、Perplexity、梯度累积、Checkpoint
  - 实验 35：TinyStories GPT 预训练
- 5.4 推理：Decoding 策略、KV Cache、Prefill/Decode
  - 实验 36：推理优化
- 5.5 SFT 与 LoRA：指令数据、Masking、参数高效微调
  - 实验 37：SFT
  - 实验 38：LoRA 对照
- 5.6 偏好优化：Reward Model、Bradley-Terry、DPO、RLHF
  - 实验 39：DPO
  - 实验 40：Reward Model
- 5.7 GRPO：组采样、相对优势、奖励归一化、Token 级优化
  - 实验 41：简化 GRPO

阶段项目：小型 LLM 训练与后训练系统。

### 第六阶段：Agentic RL（3-4 周，优先级最高）

核心问题：如何让语言模型通过与外部环境交互完成多步骤任务，并通过强化学习优化其行为？

- 6.1 Agent 与环境交互：Agent Loop、Tool Calling、轨迹、部分可观测
  - 实验 42：实现基础 Agent（Calculator / 结构化数据查询 / 受限 Python Executor）
- 6.2 轨迹与奖励：Outcome/Process Reward、可验证奖励、稀疏奖励、Reward Hacking
  - 实验 43：Agent Trajectory
  - 实验 44：Reward Design
- 6.3 Agent 策略优化：Rollout、Advantage、PPO/GRPO、KL 正则
  - 实验 45：Agent PPO / GRPO
- 6.4 综合项目：有限、可验证任务环境中的工具调用 Agent（环境定义 / Action Space / 可验证 Reward / 完整 Trajectory / 策略训练 / 独立评估 / 训练分析）

选修：Model-based RL、World Model、Planning、Multi-Agent RL。

## 四、工程能力建设

| 阶段 | 工程要求 |
|---|---|
| 概率实验 | MLX 数组与随机采样、Matplotlib（桥接）、数值统计 |
| 深度学习 | Module、自动微分、Optimizer、训练循环 |
| 生成模型 | Sampling、数值稳定性、计算图与梯度 |
| 强化学习 | Gymnasium、Rollout、Replay Buffer、环境管理 |
| LLM | 数据管线、序列处理、Checkpoint、推理优化 |
| Agentic RL | Agent Loop、Trajectory、Reward、Rollout Worker |

统一实验组织方式：`model.py` / `dataset.py` / `train.py` / `evaluate.py` / `config.py` / `metrics.py` / `checkpoint.py`。

## 五、时间与优先级

每周 10 小时初始估算，总计约 5-7 个月。时间不是硬性进度要求，每阶段必须达到验收标准才能进入下一阶段。

| 阶段 | 时间估计 | 优先级 |
|---|---|---|
| 概率建模与信息论 | 3 周 | 最高 |
| 深度学习与序列建模 | 3 周 | 高 |
| 生成模型 | 5 周 | 高 |
| 强化学习 | 5-6 周 | 最高 |
| LLM 预训练与后训练 | 5-6 周 | 最高 |
| Agentic RL | 3-4 周 | 最高 |

## 六、不纳入主线的内容

传统机器学习算法体系（SVM、决策树、随机森林等）、Offline RL、Imitation Learning、大量 CNN/GAN/Flow 变体、大型 Diffusion 训练、大型 LLM 预训练、复杂分布式训练系统、复杂 Agent 框架、World Model 深入研究。主线完成后再按需补充。

## 七、课程结构上的关键设计

1. 实验之间存在明确的知识依赖：Monte Carlo 会在 RL 阶段重新出现（策略梯度估计）；KL 会在 VAE、PPO、DPO 中反复出现。
2. 实验难度从静态数据转向动态交互：MNIST（固定数据集）→ Gymnasium（策略-环境交互）→ TinyStories（自回归生成）→ Agentic RL（多步骤环境交互）。
3. 最终项目不是简单拼接：Agentic RL 真正需要的是概率建模、策略梯度、轨迹采样、奖励设计和语言模型训练的结合。
