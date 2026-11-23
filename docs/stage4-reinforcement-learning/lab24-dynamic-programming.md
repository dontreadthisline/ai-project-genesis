---
title: "Lab 24：动态规划"
lab: 24
stage: 4
---

对应大纲实验 24。依赖 Lab 23 的 P/R 张量。模型已知（model-based 的最后一块，此后进入 model-free）。

## 0. 理论问题

1. 贝尔曼期望方程为什么有唯一解？压缩映射保证了什么？
2. 策略改进为什么单调？最优性方程的解为什么就是最优值函数？
3. 策略迭代与值迭代各以什么方式求解，收敛速度差在哪？

## 1. 数学基础

### 1.1 值函数与贝尔曼期望方程

$$
v^\pi(s) = \mathbb E_\pi\left[ G_t \mid s_t = s \right] = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\left[ r + \gamma v^\pi(s') \right]
$$

矩阵形式 $v^\pi = r^\pi + \gamma P^\pi v^\pi$，线性方程组解 $v^\pi = (I - \gamma P^\pi)^{-1} r^\pi$。迭代形式 $v_{k+1} = T^\pi v_k$ 是压缩映射：

$$
\|T^\pi u - T^\pi w\|_\infty \le \gamma \|u - w\|_\infty
$$

（逐项证明：差的每项都被 $\gamma \max |u - w|$ 控制。）Banach 不动点定理给唯一解与几何收敛，误差上界 $\|v_k - v^\pi\|_\infty \le \frac{\gamma^k}{1-\gamma}\|v_1 - v_0\|_\infty$。

### 1.2 策略改进定理（推导）

对任意 $\pi$，取 greedy 改进 $\pi'(s) = \arg\max_a q^\pi(s,a)$：

$$
v^{\pi'}(s) = q^\pi(s, \pi'(s)) = \max_a q^\pi(s,a) \ge \sum_a \pi(a|s) q^\pi(s,a) = v^\pi(s)
$$

（第一步是改进策略下反复展开，配合不等式保持。）逐状态不降，若已处处相等则 $v^\pi$ 满足最优性方程，即 $v^\pi = v^*$。

### 1.3 值迭代

$$
v_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)\left[ r + \gamma v_k(s') \right]
$$

$T^*$ 同样是 $\gamma$-压缩。策略迭代 = 交替「求 $v^\pi$（内层解方程）+ 改进」；值迭代 = 直接迭代 $T^*$。前者外层步数少但每步贵（解线性系统），后者每步便宜但步数多（几何速度同为 $\gamma$ 控制，实际常用 $\epsilon$-停机：$\|v_{k+1}-v_k\|_\infty < \epsilon(1-\gamma)/\gamma$ 保证 $\|v_k - v^*\|_\infty < \epsilon$）。

## 2. 实现任务

<div class="theorem algorithm">
<div class="theorem-title">算法 24.1（值迭代）</div>

1. **输入**：转移核 $P$、奖励 $R$、折扣 $\gamma$、容差 $\epsilon$；初始化 $v \leftarrow 0$。
2. **repeat**　对每个状态 $s$ 更新：
   $$v(s) \leftarrow \max_a \sum_{s'} P(s'\mid s,a)\left[ r + \gamma\, v(s') \right]$$
3. **until**　$\|v_{k+1} - v_k\|_\infty \le \epsilon(1-\gamma)/\gamma$（停机后 $\|v - v^*\|_\infty \le \epsilon$）。
4. **输出**：贪心策略 $\pi(s) \leftarrow \arg\max_a \sum_{s'} P(s'\mid s,a)[r + \gamma v(s')]$。

</div>

```python
def policy_evaluation(P, R, pi, gamma, eps=1e-8): ...
def policy_improvement(P, R, v, gamma): ...
def policy_iteration(P, R, gamma): ...
def value_iteration(P, R, gamma, eps=1e-8): ...
```

自测：

```python
# T1: 随机策略的迭代求值 vs 线性方程组直接解（allclose 1e-6）
# T2: 收敛速度：迭代误差比 ||v_{k+1}-v_k|| / ||v_k - v_{k-1}|| ≤ γ（slippery 下 ≈ γ）
# T3: policy_iteration 与 value_iteration 的 v* 一致（allclose 1e-6）
# T4: γ<1 时改进单调：每次改进后 v^{π'} ≥ v^π 逐状态成立
# T5: 停机界：ε-停机后 ||v-v*||∞ < ε（与更精确解对照）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $\gamma \in \{0.1, 0.5, 0.9, 0.99\}$ | — | 值迭代收敛步数（理论 $\sim \log\epsilon / \log\gamma$） |
| E2 | 策略迭代 vs 值迭代 | — | 外层迭代数、总扫描次数 |
| E3 | slippery True/False 的最优策略对照 | — | 策略差异（绕开洞的方式） |
| E4 | 各状态 $v^*$ 可视化（网格热图） | — | 距目标的梯度结构 |

## 4. 预期图表

- 图 1：$v^*$ 网格热图：从目标向外的指数衰减（$\gamma^{dist}$ 结构），洞附近值梯度被 slippery 摊平。
- 图 2：E1 收敛步数 vs $\gamma$（半对数图呈线性：$\log$ 步数 $\propto 1/|\log\gamma|$）。
- 图 3：策略迭代每轮的 $v$ 快照叠层（单调上升）。
- 图 4：两环境的最优路径叠加图（slippery 下策略偏向远离洞的方向）。

## 5. 观测指标

`v_star`、`iters_by_gamma`、`pi_star`、`convergence_ratio`（实测压缩比）。

## 6. 结果分析

1. 图 1 中 $v^*(s)$ 与「到目标最短步数 $d$」的关系是什么？写成 $v^* \approx \gamma^d$ 并对照残差。
2. E2 中两种算法的总计算量谁小？与状态数、动作数的关系如何 scale？
3. 压缩比的实测值与 $\gamma$ 的偏差来自哪里？（逐状态的局部压缩率不同，$\gamma$ 是上界。）

## 7. 验收清单

- [ ] T1-T5 通过；能白板证明 $T^\pi$ 压缩与策略改进单调
- [ ] 能解释 $\epsilon$-停机准则的来源
- [ ] 理解「动态规划需要已知 $P$」这一前提，及 Lab 25 起转向 model-free 的动机
