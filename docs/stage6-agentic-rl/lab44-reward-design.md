---
title: "Lab 44：Reward Design"
lab: 44
stage: 6
---

对应大纲实验 44。依赖 Lab 42/43。奖励是 RL 的「损失函数」：设计错误，优化必然跑偏。

## 0. 理论问题

1. Outcome / Process / Cost 三类奖励各捕获什么？冲突时如何加权？
2. Reward hacking 为什么必然发生在「代理指标 ≠ 真实目标」处？
3. 稀疏奖励与密集奖励的权衡（连 Lab 03 稀有事件）？

## 1. 数学基础

### 1.1 三类奖励

$$
R(\tau) = w_o\, \underbrace{\mathbb 1[\text{final correct}]}_{\text{outcome}} + w_p \sum_t \underbrace{s_t}_{\text{process}} - w_c\, \underbrace{K}_{\text{cost}}
$$

- Outcome：终态正确性，0/1。无偏但极稀疏（多步任务成功率低时信号稀有——Lab 03 §1.4 的稀有事件估计困难直接适用：要 $\geq 10\%$ 相对误差的组信号，组内必须有正有负）。
- Process：中间步骤的合法动作、有效工具调用、格式正确。密集但只测「行为表面」。
- Cost：步数/调用次数惩罚。防空转，过强则鼓励「跳步瞎答」。

### 1.2 Reward hacking 的结构性原因

RL 最大化的是 $R$ 而非真实目标 $R^*$；凡 $R$ 是 $R^*$ 的不完美代理，两者的差集就是可利用空间：

$$
\max_\pi \mathbb E[R] \ \neq\ \max_\pi \mathbb E[R^*] \quad \text{当 } \exists \tau:\ R(\tau) \text{ 高而 } R^*(\tau) \text{ 低}
$$

经典实例（本实验复现）：process 奖励按「工具调用次数」发 → 策略学会无限调用合法但无用的工具刷分；格式奖励按「包含某关键词」发 → 策略复读关键词。Hack 的识别信号：**代理指标上升 + 真实成功率停滞或下降**。

### 1.3 可验证性与防线

可验证奖励（程序判分）把 $R$ 与 $R^*$ 的差距压缩到判分器 bug 层面，是 Agentic RL 的主线选择。防线的层次：奖励仅绑定终态（稀疏但干净）> 绑定过程（密集但可 hack）；KL 正则限制策略偏离（Lab 39/41 的 β 机制）间接限制 hack 的可达性；成本惩罚封堵「刷次数」类 hack。

## 2. 实现任务

```python
class RewardConfig: ...     # w_outcome, w_process, w_step, w_format ...
def compute_reward(traj, cfg) -> float: ...
def make_hackable_rewards(): ...   # 故意设计有漏洞的两种奖励
```

自测：

```python
# T1: 纯 outcome：正确轨迹 R=1、错误 R=0、截断 R=0
# T2: 成本惩罚单调性：同任务两条轨迹（3 步 vs 10 步均成功），R 差 == w_c*7
# T3: hack 复现 1：process 按调用计分时，「空转刷工具」轨迹的 R 超过直接答对的最短轨迹
# T4: hack 复现 2：格式奖励含关键词时，复读关键词的轨迹得高分
# T5: 权重为零的退化：w_o=1, 其余 0 时 R == outcome（还原 Lab 43 基线）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 权重组合 $\{(1,0,0), (1, 0.1, 0.01), (0.5, 0.5, 0)\}$ | SFT 策略 | 平均奖励、真实成功率 |
| E2 | 有漏洞奖励 + 优化器（简化：贪心搜索或几步 GRPO） | — | hack 行为的出现 |
| E3 | $w_c \in \{0, 0.01, 0.1\}$ | — | 平均步数 vs 成功率 |
| E4 | outcome 稀疏度：单步/三步任务 | — | 组内有信号比例（连 Lab 41 §1.3） |

## 4. 预期图表

- 图 1：E1 的代理奖励 vs 真实成功率散点（不同权重配置）：两者解耦的证据。
- 图 2：E2 的奖励曲线双线：代理奖励上升、真实成功率水平——hack 的标准图式。
- 图 3：E3 的步数-成功率 Pareto 前沿（$w_c$ 扫描）。
- 图 4：E4 的信号率 vs 任务步数（稀疏化的定量刻画）。

## 5. 观测指标

`proxy_vs_true`、`avg_steps_by_wcost`、`signal_rate_by_depth`、`hack_cases`。

## 6. 结果分析

1. E1/E2 中你观察到哪种 hack？它利用的差集（$R$ 高 $R^*$ 低的具体路径）是什么？
2. E3 的最优 $w_c$ 如何确定？强成本惩罚的失败形态（跳步乱答）出现了吗？
3. 论证：为什么本课程最终项目用「纯 outcome + 可验证判分 + KL 正则」而非 process 奖励？

## 7. 验收清单

- [ ] T1-T5 通过；两类 hack 均被构造性复现
- [ ] 能给出 reward hacking 的结构性定义（代理-真实目标差集）
- [ ] 完成权重-步数-成功率的三维权衡分析
