---
title: "Lab 25：SARSA 与 Q-Learning"
lab: 25
stage: 4
---

对应大纲实验 25。依赖 Lab 23/24。环境：FrozenLake（slippery）+ 自制 Cliff Walking。

## 0. 理论问题

1. TD 学习用自举替代完整回合，代价与收益是什么？
2. SARSA 与 Q-Learning 只差一个更新目标，为什么学到不同策略？
3. 探索（ε-greedy）与利用的矛盾如何随训练衰减？

## 1. 数学基础

### 1.1 TD(0) 与自举

蒙特卡洛用真实回报 $G_t$ 作目标（无偏、高方差、必须等回合结束）；TD 用估计更新估计：

$$
V(s_t) \leftarrow V(s_t) + \alpha\left[ \underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD 目标}} - V(s_t) \right]
$$

TD 目标有偏（$V$ 未收敛）但方差低（只看一步转移）、无需完整回合——bias-variance 与 MC 一致（Lab 03 的主题在 RL 中的重现）。

### 1.2 两种控制更新

$$
\text{SARSA（on-policy）: } Q(s_t, a_t) \leftarrow Q + \alpha\left[ r + \gamma Q(s_{t+1}, \color{red}{a_{t+1}}) - Q(s_t,a_t) \right]
$$

$$
\text{Q-Learning（off-policy）: } Q(s_t, a_t) \leftarrow Q + \alpha\left[ r + \gamma \max_{a'} Q(s_{t+1}, \color{red}{a'}) - Q(s_t,a_t) \right]
$$

SARSA 的目标含实际执行的下一动作 $a_{t+1}$（含探索噪声），学的是「当前探索策略下的值」；Q-Learning 的目标是 greedy 最优动作，与执行策略解耦——**学的是最优策略的值，走的是探索策略的路**。Cliff Walking 的经典分化：贴崖走的 greedy 路径在探索噪声下会坠落，SARSA 学到绕远的安全路径；Q-Learning 坚持最优路径、训练期回报更低但策略本身更优。

### 1.3 收敛条件与探索

表格 TD 控制收敛需：所有 (s,a) 无限次访问 + α 满足 Robbins-Monro（$\sum\alpha = \infty, \sum\alpha^2 < \infty$）。ε-greedy 衰减（$\epsilon_t = \max(\epsilon_{min}, \epsilon_0 \cdot decay^t$）平衡探索与渐近最优。max 算子的正偏差：$\max_a \mathbb E[Q] \ge \mathbb E[\max_a Q]$（Jensen），Q-Learning 有过高估计倾向——Lab 27 在深度版中正面处理。

## 2. 实现任务

Cliff Walking：4x12 网格，底行起点左端、目标右端、中间 10 格悬崖（坠落 r = -100 回起点），每步 r = -1。

实现清单（签名与 TODO 见骨架源码）：

- `CliffWalking` — 自制环境：reset/step
- `sarsa` — sarsa
- `q_learning` — q_learning

- 接口骨架：[`../../code/lab25-sarsa-qlearning/skeleton.py`](../../code/lab25-sarsa-qlearning/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab25-sarsa-qlearning/selftest.py`](../../code/lab25-sarsa-qlearning/selftest.py)
- 实验入口：`cd code/lab25-sarsa-qlearning && uv sync && uv run run_lab25.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 算法 × 训练期回报曲线 | Cliff | 分化出现时机 |
| E2 | 学得策略的可视化路径 | — | 贴崖 vs 绕行 |
| E3 | ε 衰减速度 $\{快, 中, 慢\}$ | — | 最终回报、稳定性 |
| E4 | $\alpha \in \{0.05, 0.5\}$ | — | 收敛稳定性（大 α 震荡） |
| E5 | FrozenLake slippery 上两算法 | — | 成功率对照 |

## 4. 预期图表

- 图 1：两算法回报曲线（滑动平均）：Q-Learning 训练期更低（探索时掉崖）、渐近 SARSA 的执行回报也更稳。
- 图 2：两策略网格路径图：Q 最短贴崖、SARSA 上绕一格。
- 图 3：ε 衰减过快：回报卡在次优（探索不足）；过慢：渐近回报低（一直探索）。
- 图 4：α 大小的 loss/回报震荡形态。

## 5. 观测指标

`train_return_curve`、`final_policy_path`、`final_avg_return`、`sarsa_vs_q_return`。

## 6. 结果分析

1. 用 1.2 的两个更新目标解释图 1/2 的分化——哪个词概括 SARSA 学到的对象？（「当前策略的值」含探索风险。）
2. 图 3 的最优 ε 调度如何与「回合数、状态数」相关？
3. off-policy 有什么好处？（Q-Learning 能用任意行为策略的数据学最优策略——这是 Lab 26 Replay Buffer 的前提。）

## 7. 验收清单

- [ ] T1-T4 通过；能白板写出两种更新并说明 on/off-policy 的本质差别
- [ ] 复现 Cliff Walking 分化
- [ ] 能解释 TD 与 MC 的偏差-方差对照（连回 Lab 03）
