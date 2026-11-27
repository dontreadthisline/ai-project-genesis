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

<iframe src="resources/figures/q-overestimation.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="max 的正偏差"></iframe>

<p align="center"><b>图 27.1</b>：E[max ε] 随动作数 k</p>

### 1.2 Double DQN（解耦）

原目标：$\max_{a'} Q(s', a'; \theta^-)$——同一个网络既选出动作又评估它（选择与评估共享噪声，正相关放大偏差）。解耦：

$$
y = r + \gamma\, Q\big(s',\ \arg\max_{a'} Q(s', a'; \theta);\ \theta^-\big)
$$

在线网络 $\theta$ 选动作，目标网络 $\theta^-$ 评估：选择误差与评估误差近似独立，$\mathbb E[\text{偏差}]$ 大幅缩小（不归零）。

<iframe src="resources/figures/q-vs-return.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="高估的可视证据"></iframe>

<p align="center"><b>图 27.2</b>：Q 估计 vs 实际回报</p>

### 1.3 Dueling DQN（结构分解）

$$
Q(s, a; \theta) = V(s; \theta_V) + \left( A(s, a; \theta_A) - \frac{1}{|\mathcal A|}\sum_{a'} A(s, a'; \theta_A) \right)
$$

减均值项解决不可辨识性（$V + c,\ A - c$ 给出同一 Q）。动机：许多状态下动作选择无关紧要（值本身决定回报），dueling 让 $V$ 流从大量动作的均值中学习，$A$ 只学相对差异——动作多、多数状态「动作不敏感」时收益大。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `ddqn_target` — 1.2
- `DuelingQNet` — 共享 trunk + V head + A head

- 接口骨架：[`../../code/lab27-dqn-ablation/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab27-dqn-ablation/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab27-dqn-ablation/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab27-dqn-ablation/selftest.py)
- 实验入口：`cd code/lab27-dqn-ablation && uv sync && uv run run_lab27.py`


