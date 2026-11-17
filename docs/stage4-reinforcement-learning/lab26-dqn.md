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

### 1.3 网络与训练细节

输入 4 维状态，输出 4 个动作的 Q 值（单网多头，优于「状态入网 + 动作拼接」的玩具设计）。ε-greedy 探索，ε 从 1.0 线性衰减到 0.05。Huber 损失（$\mathcal L_\delta$：小误差二次、大误差线性）抑制离群 TD 误差的梯度爆炸。

## 2. 实现任务

```python
class QNet: ...                        # 4 -> 128 -> 128 -> 4
class ReplayBuffer: ...                # 环形缓冲：push / sample(batch)
def dqn_train(env_fn, steps, cfg): ... # ε 衰减、目标网络同步、梯度步
def eval_policy(qnet, env, episodes): ...
```

自测：

```python
# T1: Buffer 环形覆盖：容量 N 填 2N 条后，内容为后 N 条（FIFO 语义）
# T2: 目标网络滞后：软更新前后 ||θ-θ-|| 变化符合公式 τ(θ-θ-)
# T3: Huber 梯度：|err|>δ 处梯度恒为 δ（数值验证）
# T4: 训练后 Q 估计 vs 实际回报：对贪心策略，E[Q(s, a*)] 与
#     实际折扣回报均值一致（容差放宽，同一量级且 Q ≥ 回报可解释 max 偏差）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 组件逐步加入：裸 Q 网络 / +replay / +target / 全量 | — | 训练稳定性、最终回报 |
| E2 | buffer 容量 $\{1k, 50k\}$ | — | 样本新鲜度 vs 稳定性 |
| E3 | 目标网络更新频率 $\{1, 100, 1000\}$（含 1 = 无滞后） | — | 震荡与收敛速度 |
| E4 | ε 衰减速度 | — | 早期探索充分性 |

## 4. 预期图表

- 图 1：E1 四条回报曲线：裸网络剧烈震荡且常坍塌到 0（Pole 掉落 = 回报 9）；全量版本稳定升到 300+。
- 图 2：训练中 TD 误差（Huber 前的 |err|）分布演化：初期宽、后期集中于小值但保留长尾。
- 图 3：E3 三条曲线：更新太频繁震荡、太慢收敛慢。
- 图 4：E1 对应的 Q 值范数曲线：不稳定版本的 Q 范数发散性跳动。

## 5. 观测指标

`return_curve_by_ablation`、`final_eval_return`、`td_err_stats`、`q_norm_curve`。

## 6. 结果分析

1. 用 1.2 的两个机制分别解释 E1 中两步改进各消除了什么现象？
2. E4 中 Q 估计系统性高于实际回报多少？这是 max 偏差（1.3）的证据，Lab 27 正面处理。
3. replay 的 off-policy 性质使 DQN 可以用 ε-greedy 早期的高噪声数据学贪心目标——这与 Lab 25 的哪个结论呼应？

## 7. 验收清单

- [ ] T1-T4 通过；能写出 DQN 损失并解释两个组件的必要性
- [ ] 复现 E1 消融的稳定性阶梯
- [ ] CartPile eval 回报 $\geq 475$（Gymnasium 满分 500 的 95%）
