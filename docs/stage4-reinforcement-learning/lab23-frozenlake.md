---
title: "Lab 23：FrozenLake 环境建模"
lab: 23
stage: 4
---

对应大纲实验 23。环境：Gymnasium `FrozenLake4x4-v1`（ slippery 默认 True）。第四阶段的起点：先把「环境」翻译成数学对象。

## 0. 理论问题

1. 一个环境的哪些要素构成 MDP？FrozenLake 的每个要素具体是什么？
2. slippery 转移意味着什么？它如何改变问题的难度？
3. 终止状态与折扣在数学上各处理什么？

## 1. 数学基础

### 1.1 MDP 五元组

$\mathcal M = (\mathcal S, \mathcal A, P, r, \gamma)$：状态集、动作集、转移核 $P(s'|s,a)$、奖励函数 $r(s,a,s')$、折扣 $\gamma \in [0,1)$。策略 $\pi(a|s)$。轨迹：

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots), \qquad
\text{Return } G_t = \sum_{k\ge 0} \gamma^k r_{t+k+1}
$$

$\gamma$ 的三重身份：数学上保证无穷和收敛；行为上编码「未来的重要程度」（$\gamma = 0.9$ 意味着 10 步后的奖励权重只剩 $0.35$）；计算上让值函数成为贝尔曼方程的不动点。

### 1.2 FrozenLake 的实例化

- $\mathcal S = \{0..15\}$（4x4 网格），$\mathcal A = \{0..3\}$（左/下/右/上）。
- $r = 1$ 仅当进入状态 15（目标），其余 0。洞（5, 7, 11, 12）与目标为终止态。
- slippery：$P(\text{预期方向}) = 1/3$，$P(\text{垂直两侧})$ 各 $1/3$。预期动作不等于实际动作——转移核是环境的属性而非智能体的选择。

Gymnasium 通过 `env.unwrapped.P[s][a]` 暴露完整转移表：列表 of (prob, next_state, reward, done)。**本实验就是把它转成矩阵张量** $P \in \mathbb R^{S \times A \times S}$、$R \in \mathbb R^{S \times A}$，供 Lab 24/25 的表格算法直接使用。

### 1.3 转移核的合理性检查

每行检查 $\sum_{s'} P(s'|s,a) = 1$；终止态自吸收（$P(s|s,a) = 1$）；从每个非终止态出发，目标可达（连通性）。

## 2. 实现任务

```python
def build_P_R(env):
    """从 env.unwrapped.P 构造 P (S,A,S), R (S,A)，终止态自吸收处理"""
def expected_reward(R, s, a): ...
def sample_step(P, s, a, rng): ...    # 按 P 采样下一状态（自实现，不用 env.step）
```

自测：

```python
# T1: 每行概率和为 1（所有 (s,a)）
# T2: 终止态自吸收：P[hole, a, hole] == 1
# T3: slippery 概率分布：非边界状态上三个可能转移各 1/3
#     边界/角落撞墙时分布折叠（概率合并回合法状态）
# T4: sample_step 与 P 的频率一致（1e5 次采样，各 s' 频率误差 < 1%）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | slippery True / False | — | 转移矩阵的结构差异、成功概率上界 |
| E2 | $\gamma \in \{0.5, 0.9, 0.99\}$ | 固定随机策略 | 随机游走的期望回报 |
| E3 | 随机策略 Monte Carlo 回报 vs 解析期望 | — | Lab 03 估计器复用的正确性 |
| E4 | 随机策略下访问状态分布（100 回合） | — | 洞/目标的到达频率 |

## 4. 预期图表

- 图 1：网格图标注状态编号、洞、目标、每状态的转移箭头（按最大概率方向）。
- 图 2：E2 三种 $\gamma$ 的期望回报（解析 = 线性方程组的解或矩阵幂迭代）。
- 图 3：E3 收敛：MC 估计 vs 解析值，误差 $\propto 1/\sqrt n$（Lab 03 复检）。
- 图 4：E4 热力图：访问频率集中在中部路径，slippery 下更分散。

## 5. 观测指标

`P_rowsum_max_err`、`random_policy_return_by_gamma`、`mc_vs_analytic`。

## 6. 结果分析

1. slippery 下「最优策略」还能保证成功吗？期望成功率的上界受什么约束？
2. E2 中 $\gamma$ 小时为什么期望回报骤降（成功奖励在 5+ 步之外）？
3. E3 的 MC-解析对照把 Lab 03 的哪些结论直接搬过来了？

## 7. 验收清单

- [ ] T1-T4 通过；P/R 张量构建正确
- [ ] 能把任意 Gymnasium 玩具环境手工翻译成 MDP 五元组
- [ ] 能解释 slippery 对策略设计的影响
