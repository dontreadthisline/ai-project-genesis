---
title: "Lab 27：DQN 消融实验"
lab: 27
stage: 4
---

对应大纲实验 27。依赖 Lab 26。核心主题：过高估计的来源与两种结构修复。

## 0. 理论问题

1. $\max$ 算子引入的偏差在数学上是什么？噪声如何放大它？
2. Double DQN 的动作选择与评估解耦为什么能减偏？
3. Dueling 分解 $Q = V + A$ 适合什么结构的环境？

## 1. 数学基础

### 1.1 过高估计的来源（推导）

设各动作的真值 $Q^*(s, a_i) = Q$ 相同，估计含独立零均值噪声 $\epsilon_i$。Q-Learning 目标取 $\max_i (Q + \epsilon_i) = Q + \max_i \epsilon_i$：

$$
\mathbb E\left[\max_i \epsilon_i\right] > 0 \quad (\text{Jensen: } \max \text{ 凸})
$$

且随动作数与噪声方差增大而增大。自举把高估写回 Q 表，逐轮放大。数值感受：$\epsilon_i \sim \mathcal N(0, 1)$、4 个动作时 $\mathbb E[\max] \approx 1.03$——每步自举平均抬升一个标准差的量级。

### 1.2 Double DQN（解耦）

原目标：$\max_{a'} Q(s', a'; \theta^-)$——同一个网络既选出动作又评估它（选择与评估共享噪声，正相关放大偏差）。解耦：

$$
y = r + \gamma\, Q\big(s',\ \arg\max_{a'} Q(s', a'; \theta);\ \theta^-\big)
$$

在线网络 $\theta$ 选动作，目标网络 $\theta^-$ 评估：选择误差与评估误差近似独立，$\mathbb E[\text{偏差}]$ 大幅缩小（不归零）。

### 1.3 Dueling DQN（结构分解）

$$
Q(s, a; \theta) = V(s; \theta_V) + \left( A(s, a; \theta_A) - \frac{1}{|\mathcal A|}\sum_{a'} A(s, a'; \theta_A) \right)
$$

减均值项解决不可辨识性（$V + c,\ A - c$ 给出同一 Q）。动机：许多状态下动作选择无关紧要（值本身决定回报），dueling 让 $V$ 流从大量动作的均值中学习，$A$ 只学相对差异——动作多、多数状态「动作不敏感」时收益大。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `ddqn_target` — 1.2
- `DuelingQNet` — 共享 trunk + V head + A head

- 接口骨架：[`../../code/lab27-dqn-ablation/skeleton.py`](../../code/lab27-dqn-ablation/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab27-dqn-ablation/selftest.py`](../../code/lab27-dqn-ablation/selftest.py)
- 实验入口：`cd code/lab27-dqn-ablation && uv sync && uv run run_lab27.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | DQN vs Double DQN（CartPole） | 同种子组 | Q 估计 vs 实际回报的偏差 |
| E2 | 随机网络的高估测量：$\mathbb E[\max_a Q]$ vs $\mathbb E[Q]$ | — | 偏差直接观测 |
| E3 | Dueling vs 普通（CartPole） | — | 收敛速度（动作少时差距小） |
| E4 | 偏差量与训练稳定性的相关（散点） | 全部 run | 高估 ↔ 坍塌 |

## 4. 预期图表

- 图 1：T1 的 E[max] vs k 曲线（叠加理论近似 $\sqrt{2\ln k}\,\sigma$）。
- 图 2：训练全程的 $\bar Q(s, a^*)$ 与实际回报均值双线：DQN 的 Q 线系统性高于回报线，Double 版两者贴合。
- 图 3：E3 两条学习曲线。
- 图 4：E4 散点 + 拟合：高估幅度大的 run 更早坍塌。

## 5. 观测指标

`q_overestimation`（Q 均值 - 回报均值）、`max_bias_by_k`、`return_curves`。

## 6. 结果分析

1. 图 2 的偏差差值多大？与 T1 的理论量级一致吗？
2. 为什么 Double 不需要额外网络参数也能减偏？（只用现有两网络的分工变化。）
3. Dueling 在 4 动作 CartPole 上优势是否可见？预测什么环境会放大它的优势？

## 7. 验收清单

- [ ] T1-T4 通过；能白板推导 max 偏差并解释 Double 的解耦逻辑
- [ ] 复现 Q 高估曲线与 Double 的修复
- [ ] 能说明 Dueling 的适用条件
