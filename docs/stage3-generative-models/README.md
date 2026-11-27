---
title: "第3部分：生成模型"
part: 3
---

核心问题：如何让神经网络学习数据分布，并从学习到的分布中生成新样本？



## 实验列表

| 实验 | 主题 | 知识块 |
|---|---|---|
| 实验 15 | 实现 VAE（二维潜变量） | 潜变量模型、变分推断、ELBO、重参数化 |
| 实验 16 | Reparameterization 与梯度传播 | 随机节点、确定性路径 |
| 实验 17 | 实现基础 GAN | 极小极大博弈、JS 散度、Mode Collapse |
| 实验 18 | 二维 Normalizing Flow | 变量替换、可逆变换、Jacobian 行列式 |
| 实验 19 | 实现 DDPM | 前向马尔可夫过程、噪声调度、反向去噪 |
| 实验 20 | 二维 Flow Matching | 概率路径、向量场、ODE |
| 实验 21 | MNIST Flow Matching | 采样轨迹、积分步数与质量 |
| 实验 22 | Diffusion 与 Flow Matching 对照 | 训练目标、概率路径、采样过程 |

四个范式的知识内容：

1. **VAE：潜变量概率建模**：Latent Variable Model、Encoder/Decoder、Approximate Posterior、Variational Inference、ELBO、Reparameterization Trick、KL Regularization。
2. **GAN：对抗式分布学习**：Generator、Discriminator、Minimax Game、Nash Equilibrium、JS Divergence、Mode Collapse、Training Instability。
3. **Normalizing Flow：可逆概率变换**：Change of Variables、Invertible Transformation、Jacobian Determinant、Exact Likelihood（本阶段只做二维直觉实验，不展开复杂架构）。
4. **Diffusion 与 Flow Matching：逐步去噪与连续概率流**：Forward Markov Process、Noise Schedule、Marginal Distribution、Reverse Process、Noise Prediction、Score Function、DDPM/DDIM、Probability Flow ODE、Conditional Probability Path、Vector Field、Rectified Flow。

## 概念接口

**对本阶段的前置依赖**：

- 实验 15 的 ELBO 中 KL 项 = Lab 04 的高斯对高斯闭式解；重构项 = Lab 03 的期望估计。
- 实验 16 的重参数化与 Lab 01 的 Cholesky 采样（$x = \mu + Lz$）是同一思想：把随机性外移，让梯度穿过确定性路径。
- 实验 19 的 $q(x_t \mid x_0)$ 与反向条件 $p(x_{t-1} \mid x_t)$ 是 Lab 02 条件分布链的直接应用；噪声预测目标即条件期望。

**后续阶段的复用**：

- 实验 20/21 的速度场积分器与第五阶段解码采样（实验 36）在数值上同构。
- 本阶段建立的「生成建模 = 从噪声到数据的分布变换」视角，是理解 LLM 采样多样性（temperature/top-p）的背景。

## 工程要求

Sampling、数值稳定性（log-sum-exp、log 空间计算）、计算图与梯度管理。训练基座直接复用第二阶段框架。

## 阶段验收

能够解释 VAE、GAN、Flow、Diffusion、Flow Matching 的核心数学机制，并独立实现其中的主要模型（VAE、GAN、简化 DDPM、简化 Flow Matching 为必做，复杂 Flow 架构不作要求）。

## 阶段项目

## 概念接口

**前置依赖**：实验 15 的 ELBO 中 KL 项 = Lab 04 的高斯对高斯闭式解；重构项 = Lab 03 的期望估计。 实验 16 的重参数化与 Lab 01 的 Cholesky 采样（$x = \mu + Lz$）是同一思想：把随机性外移，让梯度穿过确定性路径。 实验 19 的 $q(x_t \mid x_0)$ 与反向条件 $p(x_{t-1} \mid x_t)$ 是 Lab 02 条件分布链的直接应用；噪声预测目标即条件期望。

**后续复用**：实验 20/21 的速度场积分器与第五阶段解码采样（实验 36）在数值上同构。
本阶段建立的「生成建模 = 从噪声到数据的分布变换」视角，是理解 LLM 采样多样性（temperature/top-p）的背景。

## 工程要求

Sampling、数值稳定性（log-sum-exp、log 空间计算）、计算图与梯度管理。训练基座直接复用第二阶段框架。

## 阶段验收

能够解释 VAE、GAN、Flow、Diffusion、Flow Matching 的核心数学机制，并独立实现其中的主要模型（VAE、GAN、简化 DDPM、简化 Flow Matching 为必做，复杂 Flow 架构不作要求）。

## 阶段项目

MNIST 生成模型实验库：统一训练、采样、可视化和指标记录接口，四种范式共用同一套数据管线与评估工具（FID 不作要求，以可视化与似然/损失曲线为主）。
