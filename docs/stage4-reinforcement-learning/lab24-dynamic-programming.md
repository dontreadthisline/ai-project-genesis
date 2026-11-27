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

实现清单（签名与 TODO 见骨架源码）：

- `policy_evaluation` — policy_evaluation
- `policy_improvement` — policy_improvement
- `policy_iteration` — policy_iteration
- `value_iteration` — value_iteration

- 接口骨架：[`../../code/lab24-dynamic-programming/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab24-dynamic-programming/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab24-dynamic-programming/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab24-dynamic-programming/selftest.py)
- 实验入口：`cd code/lab24-dynamic-programming && uv sync && uv run run_lab24.py`


