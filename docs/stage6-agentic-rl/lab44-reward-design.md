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

<iframe src="resources/figures/proxy-vs-true.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="Reward Hacking 的标准图式"></iframe>

<p align="center"><b>图 44.1</b>：代理分 ↑ 而真实成功率 →</p>

### 1.3 可验证性与防线

可验证奖励（程序判分）把 $R$ 与 $R^*$ 的差距压缩到判分器 bug 层面，是 Agentic RL 的主线选择。防线的层次：奖励仅绑定终态（稀疏但干净）> 绑定过程（密集但可 hack）；KL 正则限制策略偏离（Lab 39/41 的 β 机制）间接限制 hack 的可达性；成本惩罚封堵「刷次数」类 hack。

<iframe src="resources/figures/weight-pareto.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="成本惩罚的 Pareto 前沿"></iframe>

<p align="center"><b>图 44.2</b>：成功率 vs 平均步数（w_c 扫描）</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `RewardConfig` — w_outcome, w_process, w_step, w_format ...
- `compute_reward` — 故意设计有漏洞的两种奖励
- `make_hackable_rewards` — 故意设计有漏洞的两种奖励

- 接口骨架：[`../../code/lab44-reward-design/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab44-reward-design/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab44-reward-design/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab44-reward-design/selftest.py)
- 实验入口：`cd code/lab44-reward-design && uv sync && uv run run_lab44.py`


