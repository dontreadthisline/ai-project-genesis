---
title: "Lab 26：实现 DQN"
lab: 26
stage: 4
---

对应大纲实验 26。环境：Gymnasium CartPole-v1。依赖 Lab 09 网络、Lab 25 Q-Learning。

## 0. 理论问题

1. 表格 Q 到函数近似 Q(s,a;θ)，优化目标变成了什么？
2. Replay Buffer 和 Target Network 各自解决什么病？
3. 「致命三要素」为什么组合起来危险？

## 1. 数学基础

### 1.1 从表格到可微目标

表格 Q-Learning 逐条目更新；函数近似把目标写成回归损失：

$$
\mathcal L(\theta) = \mathbb E_{(s,a,r,s') \sim \mathcal D}\left[ \left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta^-)}_{\text{目标 } y} - Q(s, a; \theta) \right)^2 \right]
$$

$\theta^-$ 是目标网络参数（滞后副本），$\mathcal D$ 是 replay 分布。

### 1.2 死亡三要素与两个解药

函数近似 + 自举 + off-policy 组合下无收敛保证（致命三要素）。两个组件各自对症：

**Replay Buffer**：打破样本时间相关性（i.i.d. 假设是 SGD 收敛分析的前提，连败 100 个相似样本会把 Q 推向极端）；提高样本复用（一次交互多次学习）；固定数据分布使自举目标相对稳定（半梯度分析的前提）。

**Target Network**：目标 $y$ 若用在线 $\theta$，则 $\theta$ 更新 → 目标移动 → 追逐自己的尾巴（复合误差，贝尔曼残差被不断放大）。$\theta^-$ 每 $C$ 步或按 $\tau \theta + (1-\tau)\theta^-$ 软更新，把目标短期冻结。

<iframe src="resources/figures/replay-target.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="两个解药各治一病"></iframe>

<p align="center"><b>图 26.1</b>：解药与病灶的对应</p>

### 1.3 网络与训练细节

输入 4 维状态，输出 4 个动作的 Q 值（单网多头，优于「状态入网 + 动作拼接」的玩具设计）。ε-greedy 探索，ε 从 1.0 线性衰减到 0.05。Huber 损失（$\mathcal L_\delta$：小误差二次、大误差线性）抑制离群 TD 误差的梯度爆炸。

<iframe src="resources/figures/component-ablation.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="组件的阶梯"></iframe>

<p align="center"><b>图 26.2</b>：逐步加组件的回报曲线</p>

## 2. 实现任务

<div class="theorem algorithm">
<div class="theorem-title">算法 26.1（DQN 训练循环）</div>

1. **初始化**：在线网络 $Q_\theta$；目标网络 $\theta^- \leftarrow \theta$；空回放缓冲 $\mathcal D$。
2. **for** $t = 1, \dots, N$ **：**
   1. $\epsilon$-greedy 采样 $a_t$，执行得 $(s_t, a_t, r_t, s_{t+1}, d_t)$，存入 $\mathcal D$。
   2. 从 $\mathcal D$ 均匀采样 minibatch $B$。
   3. 构造目标 $y = r + \gamma (1 - d)\max_{a'} Q_{\theta^-}(s', a')$。
   4. Huber 损失 $L(\theta) = \mathbb E_B\left[ \delta_\rho\!\left(y - Q_\theta(s, a)\right) \right]$，梯度步更新 $\theta$。
   5. 每 $C$ 步同步 $\theta^- \leftarrow \theta$。

</div>

实现清单（签名与 TODO 见骨架源码）：

- `QNet` — 4 -> 128 -> 128 -> 4
- `ReplayBuffer` — 环形缓冲：push / sample(batch)
- `dqn_train` — ε 衰减、目标网络同步、梯度步
- `eval_policy` — eval_policy

- 接口骨架：[`../../code/lab26-dqn/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab26-dqn/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab26-dqn/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab26-dqn/selftest.py)
- 实验入口：`cd code/lab26-dqn && uv sync && uv run run_lab26.py`


