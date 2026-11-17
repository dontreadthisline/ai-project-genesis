---
title: "Lab 29：Actor-Critic"
lab: 29
stage: 4
---

对应大纲实验 29。环境：CartPole。依赖 Lab 28。

## 0. 理论问题

1. 用学到的 $V^\pi$ 做 baseline，比批内统计好在哪里？
2. TD 误差形式的 advantage 兼顾了什么？
3. Actor 和 Critic 的学习率、更新频率为什么需要平衡？

## 1. 数学基础

### 1.1 优势函数

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s), \qquad \mathbb E_{a\sim\pi}[A^\pi(s,a)] = 0
$$

策略梯度的优势形式：

$$
\nabla_\theta J = \mathbb E\left[ \sum_t A^\pi(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$

（由 $Q = A + V$ 与 baseline 无偏性直接得到。）直觉：$A > 0$ 的动作比该状态下平均更好，提高其概率；$A < 0$ 降低。逐状态去中心比全局 baseline 更精细。

### 1.2 TD 误差作为优势的估计

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t), \qquad \mathbb E_{\pi}[\delta_t \mid s_t, a_t] = Q^\pi(s_t,a_t) - V^\pi(s_t) = A^\pi(s_t, a_t)
$$

（第二式用 $V$ 的贝尔曼方程展开即得。）$\delta_t$ 只需一次 $V$ 前向（而非两条轨迹），是优势的单步无偏（给定当前 $V$）估计——方差比 $G_t - V$ 低，代价是引入 $V$ 估计的偏差。Critic 损失：

$$
\mathcal L_V = \mathbb E\left[ (r + \gamma V(s'; \phi) - V(s; \phi))^2 \right] \quad (\text{同 Lab 26 的 target 病，可加目标网络或小 } \tau)
$$

### 1.3 联合训练

Actor 用 $-\log\pi \cdot \delta.\text{detach}()$，critic 用 $\delta^2$。两者耦合的非平稳性：critic 落后则 actor 用错误优势，actor 快变则 critic 追不上分布。实践：critic lr 常大于 actor lr、或 critic 每步多次更新。

## 2. 实现任务

```python
class ActorCritic: ...     # 共享 trunk：策略头 logits + 价值头标量
def a2c_train(env_fn, steps, cfg): ...   # on-policy：采集 n_step 回合批，一次更新
```

自测：

```python
# T1: δ 的期望 = 优势：表格 MDP（Lab 23 的 P/R）上，收敛 V 后验证
#     E_{a}[δ(s,a)] ≈ 0（均值）且 E[δ | s, a*] > E[δ | s, a_worst]
# T2: detach 检查：actor 损失对 V 参数梯度恒为 0
# T3: A 的符号逻辑：构造两条动作轨迹（好坏），δ 符号与动作质量一致
# T4: 训练后 V(s) 与实际折扣回报的 MC 估计一致（容差 15%）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | REINFORCE($G_t$-baseline) vs A2C(δ) | 同种子组 | 收敛速度、稳定性 |
| E2 | critic lr / actor lr $\in \{0.1×, 1×, 10×\}$ | — | 两头各自的收敛 |
| E3 | n-step 回传 $n \in \{1, 5, 20\}$ | — | 偏差-方差（连 Lab 25 的 TD/MC 光谱） |
| E4 | 共享 trunk vs 分离网络 | — | 样本效率与干扰 |

## 4. 预期图表

- 图 1：两条回报曲线：A2C 早期更快（低方差优势信号），长程与调好的 REINFORCE 相当。
- 图 2：E3 的 $n$ 光谱：n=1 最稳但略慢，n=20 接近 MC 的方差。
- 图 3：V 学习曲线：critic 损失与回报同步收敛。
- 图 4：E2 失衡示例：actor lr 过大时策略震荡、critic 跟不上（V 损失发散）。

## 5. 观测指标

`return_curves`、`critic_loss_curve`、`v_vs_mc`、`curves_by_nstep`。

## 6. 结果分析

1. T4 的 V-MC 一致性误差有多大？来源（函数近似 + 非平稳策略）？
2. E2 的失衡模式说明联合优化的什么本质困难？
3. on-policy 的 A2C 用完即丢数据——与 Lab 26 replay 的对比引出什么改进方向？（重要性采样，Lab 30。）

## 7. 验收清单

- [ ] T1-T4 通过；能推导 $\mathbb E[\delta] = A$ 并写出两头损失
- [ ] 复现 A2C 相对 REINFORCE 的稳定性优势
- [ ] 能叙述策略方法改进链：REINFORCE → baseline → AC → GAE/（PPO）
