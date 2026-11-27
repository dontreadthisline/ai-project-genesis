---
title: "Lab 39：DPO"
lab: 39
stage: 5
---

对应大纲实验 39。依赖 Lab 37 的 SFT 模型（作为 reference 与初始 policy）。Lab 04 的 KL 在此第三次出现（VAE、PPO 之后）。

## 0. 理论问题

1. 偏好数据 $(x, y_w, y_l)$ 比「打分数据」好在哪？Bradley-Terry 把偏好转化成什么？
2. DPO 如何绕过显式 reward model 和 RL？
3. $\beta$ 控制什么？极端值各发生什么？

## 1. 数学基础

### 1.1 Bradley-Terry 模型

偏好概率由奖励差决定：

$$
p(y_w \succ y_l \mid x) = \sigma\big( r(x, y_w) - r(x, y_l) \big)
$$

偏好数据只要求「相对好」，BT 把相对判断映射到标量差——reward model 的训练目标（Lab 40）与此相同。

### 1.2 从 RLHF 到 DPO（推导骨架）

RLHF 的 KL 约束优化问题有闭式解（对每 prompt 独立）：

$$
\pi^*(y|x) \propto \pi_{ref}(y|x)\, \exp\!\big( r(x,y) / \beta \big)
$$

反解出隐式奖励：

$$
r(x, y) = \beta \log\frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

代入 BT（$Z(x)$ 在好坏对的差中被消去）：

$$
p(y_w \succ y_l) = \sigma\left( \beta \log\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

**损失直接作用在策略上**：

$$
\mathcal L_{DPO} = -\mathbb E\left[ \log \sigma\left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

### 1.3 梯度方向

$$
\nabla \mathcal L \propto -\underbrace{\sigma(\hat r_l - \hat r_w)}_{\text{错判程度}}\Big[ \nabla\log\pi(y_w) \cdot \tfrac{1}{|y_w|} - \nabla\log\pi(y_l) \cdot \tfrac{1}{|y_l|} \Big]
$$

隐式奖励差已正确时梯度趋零；否则按错误概率加权提高 $y_w$、压低 $y_l$ 的对数概率（按长度归一）。$\beta$ 小 → 约束弱，策略可大幅偏离 ref（过拟合偏好噪声）；$\beta$ 大 → 几乎不动。

### 1.4 隐式奖励的 KL 视角

$\log\frac{\pi}{\pi_{ref}}$ 的期望正是 $-D_{KL}(\pi \| \pi_{ref})$ 的被积项（Lab 04 §1.3 反向 KL）：DPO 在逐 token 的对数比空间做梯度下降，等价于控制策略与 reference 的反向 KL 散度——与 PPO 的 KL 惩罚（Lab 30 §1.4）殊途同归。

## 2. 实现任务

偏好数据构造（可验证）：算术任务中正确回答为 $y_w$、错误为 $y_l$（多种错误类型：算错/格式错/胡说）；一部分 $y_w$ 用 SFT 模型采样生成保证 on-distribution。

实现清单（签名与 TODO 见骨架源码）：

- `dpo_loss` — 逐 token logprob 求和（mask）
- `get_logps` — 逐 token logprob 求和（mask）
- `dpo_train` — dpo_train

- 接口骨架：[`../../code/lab39-dpo/skeleton.py`](../../code/lab39-dpo/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab39-dpo/selftest.py`](../../code/lab39-dpo/selftest.py)
- 实验入口：`cd code/lab39-dpo && uv sync && uv run run_lab39.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $\beta \in \{0.05, 0.1, 0.5, 1.0\}$ | 同数据 | 偏好准确率、与 ref 的 KL |
| E2 | 训练前后的生成分布 | — | 正确率提升 + 退化形态 |
| E3 | 隐式奖励 vs 真实正确性 | — | 相关性（散点） |
| E4 | DPO vs SFT-only 对照 | — | 任务准确率、回复风格 |

## 4. 预期图表

- 图 1：E1 的偏好准确率与 KL 双轴：β 小准确率高但 KL 大（偏离 ref 越远）。
- 图 2：训练前后的隐式奖励分布：$r̂_w$ 右移、$r̂_l$ 左移、可分性增强。
- 图 3：E3 散点：隐式奖励与「是否正确」的相关（构造任务上应显著）。
- 图 4：E4 样例对照：DPO 后错误答案频率下降、格式更稳。

## 5. 观测指标

`dpo_loss_curve`、`preference_acc`、`kl_to_ref`、`task_acc_before_after`。

## 6. 结果分析

1. E1 中 KL 随 β 的变化验证了 1.3 的哪个论断？
2. DPO 训练的是「差」——它能否保证 $y_w$ 的绝对质量？（否：压低 $y_l$ 同样减小损失。观察是否有「双降」样本。）
3. 为什么 DPO 需要 SFT 起点而非基座？（$\pi_{ref}$ 决定可行域；基座不会按指令作答，偏好对不可比。）

## 7. 验收清单

- [ ] T1-T4 通过；能写出 DPO 损失与隐式奖励的反解推导
- [ ] 能解释 β 与 KL 的制衡及选择逻辑
- [ ] 构造任务上偏好准确率 > 85% 且任务正确率提升
