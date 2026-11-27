---
title: "Lab 45：Agent PPO / GRPO"
lab: 45
stage: 6
---

对应大纲实验 45 + 综合项目。依赖前两个阶段全部：Lab 41 的 GRPO 机制、Lab 42 的环境、Lab 43 的轨迹、Lab 44 的奖励。

## 0. 理论问题

1. 把 GRPO 从「单轮生成」迁移到「多步工具交互」，轨迹分布发生了什么变化？
2. 组内基线在多步任务中还够用吗？
3. 训练后的改善具体发生在哪些行为维度（连 Lab 43 的失败分类树）？

## 1. 数学基础

### 1.1 轨迹分布的目标

$$
J(\theta) = \mathbb E_{\tau \sim \pi_\theta}\!\left[ R(\tau) \right], \qquad
\tau \sim p(x)\prod_{t=1}^{K} \pi_\theta(a_t \mid h_t)\, \mathbb 1[\text{env 确定性}]
$$

与 Lab 41 的差别：一条「样本」从一次生成变为整段多步交互；对数概率是各动作段 token 的联合（观测段的概率不属于策略，条件化掉）：

$$
\log \pi_\theta(\tau \mid x) = \sum_{t=1}^{K} \sum_{j} \log \pi_\theta(a_{t,j} \mid h_t, a_{t,<j})
$$

### 1.2 组相对优势（多步版）

同 prompt 组：$G$ 条完整轨迹 $\tau_1..\tau_G$，$A_i = \frac{R_i - \mathrm{mean}}{\mathrm{std}}$（Lab 41 §1.1 原样），广播到 $\tau_i$ 的全部动作 token。信用分配是轨迹级的：整条对/错，所有动作 token 同权承担——多步信用分配的粗糙近似（精化方向是过程奖励/树搜索，超出主线）。

token 级目标与 KL 项同 Lab 41 §1.1，$\rho_{t,j}$ 逐动作段 token 计。

### 1.3 On-policy 数据的工程约束

rollout 必须用当前（或上一轮）策略采样；轨迹数据用后即弃（off-policy 偏差在多步下被复合放大）。批量 rollout 的吞吐决定训练速度：环境步（工具执行）与模型步（解码）交替，串行慢——并行 worker（独立 env + 批量解码）是必要的工程结构。

<iframe src="resources/figures/seven-steps.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="训练闭环的七步"></iframe>

<p align="center"><b>图 45.1</b>：on-policy 七步环</p>

### 1.4 训练-评估协议

- 指标：真实成功率（不含 process 奖励）、动作合法率、平均步数、无效动作率、KL to ref、截断率——训练前后按 Lab 43 分类树逐项对照；
- 早停与回滚：成功率不升或 KL 爆炸时回滚 checkpoint（Lab 44 的 hack 监控持续在线）。

<iframe src="resources/figures/before-after.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="训练改变的是什么"></iframe>

<p align="center"><b>图 45.2</b>：训练前后指标对照</p>

## 2. 实现任务

完整闭环七步（大纲 6.3）：

实现清单（签名与 TODO 见骨架源码）：

- `collect_group_rollouts` — 1.prompt 采样 2.agent rollout 3.轨迹收集（含逐步 logprob）
- `compute_group_reward` — 4.奖励计算
- `compute_advantages` — 5.组归一化
- `grpo_update` — 6.token 级更新
- `evaluate` — 7.独立评估（真实成功率）
- `train_loop` — 以上循环 + KL 监控 + hack 监控 + checkpoint

- 接口骨架：[`../../code/lab45-agent-rl/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab45-agent-rl/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab45-agent-rl/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab45-agent-rl/selftest.py)
- 实验入口：`cd code/lab45-agent-rl && uv sync && uv run run_lab45.py`


## 3. 课程终点

至此六阶段闭环：从 Lab 01 的高斯密度到本实验的多步工具交互 RL，全部数学与工程链条贯通。选修方向（Model-based RL、World Model、Planning、Multi-Agent）以当前能力为起点，按需展开。
