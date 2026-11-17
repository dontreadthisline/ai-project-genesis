---
title: "附录 A：数学符号表"
appendix: true
---

全书统一使用的符号约定。按主题分组，「首见」指该符号第一次被正式引入的实验。

## 概率与信息论（第一部分）

| 符号 | 含义 | 首见 |
|---|---|---|
| $p(x)$, $P$ | 密度函数；分布 | Lab 01 |
| $\mathbb E[f(X)]$ | 数学期望 | Lab 03 |
| $\mathrm{Var}$, $\sigma^2$ | 方差 | Lab 01 |
| $\mu$, $\Sigma$ | 均值向量；协方差矩阵 | Lab 01 |
| $\rho$ | 相关系数 | Lab 01 |
| $\mathcal N(\mu, \sigma^2)$ | 高斯分布 | Lab 01 |
| $\hat\theta$ | 参数 $\theta$ 的估计量 | Lab 05 |
| $H(P)$, $H(P,Q)$ | 熵；交叉熵（nats） | Lab 04 |
| $D_{\mathrm{KL}}(P \| Q)$ | KL 散度 | Lab 04 |
| $L(\theta)$, $\ell(\theta)$ | 似然；对数似然 | Lab 05 |

## 深度学习（第二部分）

| 符号 | 含义 | 首见 |
|---|---|---|
| $W$, $b$ | 权重矩阵；偏置 | Lab 06 |
| $z$, $h$ | 前向传播的中间量 | Lab 06 |
| $\mathcal L$ | 损失函数 | Lab 06 |
| $\bar{(\cdot)}$ | 反向传播中的上游梯度 | Lab 06 |
| $g_t$ | 梯度 | Lab 07 |
| $\eta$ | 学习率 | Lab 07 |
| $\beta_1, \beta_2$ | Adam 一阶/二阶矩衰减率 | Lab 07 |
| $\lambda$ | 权重衰减系数 | Lab 07 |
| $K$ | 卷积核尺寸 | Lab 09 |
| $\mathrm{RF}$ | 感受野 | Lab 09 |

## 生成模型（第三部分）

| 符号 | 含义 | 首见 |
|---|---|---|
| $z$ | 潜变量 / 噪声源 | Lab 15 |
| $q_\phi(z\|x)$ | 近似后验 | Lab 15 |
| $\beta$（VAE） | 重构-KL 权衡系数 | Lab 15 |
| $D$, $G$ | 判别器；生成器 | Lab 17 |
| $f$, $f^{-1}$ | 可逆变换及其逆 | Lab 18 |
| $x_t$, $t$, $T$ | 扩散过程的中间变量、时刻、总步数 | Lab 19 |
| $\beta_t$, $\bar\alpha_t$ | 噪声调度；累积调度 $\prod_{s\le t}(1-\beta_s)$ | Lab 19 |
| $\epsilon$, $\epsilon_\theta$ | 噪声；噪声预测网络 | Lab 19 |
| $v_t$ | 速度场 | Lab 20 |
| $u_t$ | 条件速度场 | Lab 20 |

## 强化学习（第四部分）

| 符号 | 含义 | 首见 |
|---|---|---|
| $s$, $a$, $r$ | 状态、动作、奖励 | Lab 23 |
| $P(s'\|s,a)$ | 转移核 | Lab 23 |
| $\gamma$ | 折扣因子 | Lab 23 |
| $\pi(a\|s)$ | 策略 | Lab 23 |
| $G_t$ | 回报 $\sum_{k\ge0}\gamma^k r_{t+k+1}$ | Lab 23 |
| $v^\pi(s)$, $q^\pi(s,a)$ | 状态值；动作值 | Lab 24 |
| $\delta_t$ | TD 误差 | Lab 25 |
| $\epsilon$（探索） | $\epsilon$-greedy 探索率 | Lab 25 |
| $\alpha$（学习率） | TD/策略更新步长 | Lab 25 |
| $A(s,a)$ | 优势函数 | Lab 29 |
| $\lambda$（GAE） | 偏差-方差插值系数 | Lab 30 |
| $\rho_t(\theta)$ | 重要性采样比 $\pi_\theta/\pi_{old}$ | Lab 30 |
| $\mathcal H(\pi)$ | 策略熵 | Lab 31 |
| $\alpha$（SAC） | 熵温度系数 | Lab 31 |

## 语言模型与后训练（第五、六部分）

| 符号 | 含义 | 首见 |
|---|---|---|
| $x_t$, $y$ | token 序列元素；回复 | Lab 33 |
| $V$ | 词表大小 | Lab 32 |
| $\mathcal T$ | 分词器 | Lab 32 |
| $T$ | 序列长度 / 上下文长度 | Lab 33 |
| $\mathrm{PPL}$ | 困惑度 $e^{\mathcal L}$ | Lab 35 |
| $\pi_{ref}$ | 参考模型（SFT 起点） | Lab 39 |
| $\beta$（DPO/GRPO） | KL 约束强度 | Lab 39 |
| $r_\phi(x, y)$ | 奖励模型 | Lab 40 |
| $G$（GRPO） | 组采样大小 | Lab 41 |
| $R(\tau)$ | 轨迹总奖励 | Lab 45 |
| $\tau$ | （交互）轨迹 | Lab 43 |

同名异义对照：$\beta$ 在 VAE（权衡）、扩散（噪声调度）、DPO/GRPO（约束强度）、Adam（矩衰减）中含义不同；$\alpha$ 在 TD 学习（步长）与 SAC（温度）中含义不同；$\lambda$ 在 GAE 与权重衰减中含义不同。以上按所属上下文区分，各章首次使用处均有定义。
