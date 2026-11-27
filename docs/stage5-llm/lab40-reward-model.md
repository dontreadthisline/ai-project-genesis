---
title: "Lab 40：Reward Model"
lab: 40
stage: 5
---

对应大纲实验 40。依赖 Lab 34 架构、Lab 39 的偏好数据构造。

## 0. 理论问题

1. 从「数值奖励」到「偏好判断」，标注为什么更可靠？BT 如何利用这一点？
2. RM 的输出是什么空间？为什么不用回归直接拟合分数？
3. RM 的失效模式（过拟合偏好表面）如何检验？

## 1. 数学基础

### 1.1 BT 训练目标

RM $r_\phi(x, y)$：LM 骨干 + 标量头。偏好对损失：

$$
\mathcal L_{RM} = -\mathbb E_{(x, y_w, y_l)}\left[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \right]
$$

注意与 DPO 的形态对照：DPO 的「奖励」是策略对数比（模型自身），RM 的奖励是独立网络——RLHF 走 RM + PPO 路线，DPO 把两者合一。

梯度：$\nabla \propto -\sigma(r_l - r_w)(\nabla r_w - \nabla r_l)$——分不开的对拉大间隔，已分开的几乎不更新（对比 margin-based 损失的常数梯度）。

<iframe src="resources/figures/rm-separation.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="Reward Model 的判别"></iframe>

<p align="center"><b>图 40.1</b>：rw 与 rl 的分数分布</p>

### 1.2 为什么偏好而非打分

绝对分数标注跨标注者不可比（每人「7 分」含义不同，Lab 04 的交叉熵视角：分数不是概率）；两两比较在标注者内部一致性好。BT 把比较还原为共享的潜在标量：只利用了「序」的信息，天然丢弃不可比的水平差异。

### 1.3 RM 的检验协议

- 判别准确率：held-out 对上的 $r_w > r_l$ 比例；
- 与真值的相关：构造任务上真值（对错）已知，RM 分数应与之一致（AUC）；
- 泛化：对未见过的错误类型（构造时故意留出）的判别力——RM 过拟合表面特征（长度、格式）时会在此暴露。

<iframe src="resources/figures/length-bias.html" style="width:100%;height:354px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="RM 的表面特征依赖"></iframe>

<p align="center"><b>图 40.2</b>：分数随长度的漂移</p>

### 1.4 RM 与可验证奖励的边界

算术/格式任务的奖励可以程序判定（verifiable reward，Lab 41/44 的主线）；RM 的价值在「不可程序化」的质量（通顺、有用）。本实验用可验证任务训练 RM 是受控研究：真值已知，RM 学到的近似与真值的差距可量化。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `RewardModel` — GPT 骨干 + scalar head，最后一个 token 位置的输出
- `rm_loss` — 1.1 公式
- `rm_train` — 判别 AUC
- `rm_auc` — 判别 AUC

- 接口骨架：[`../../code/lab40-reward-model/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab40-reward-model/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab40-reward-model/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab40-reward-model/selftest.py)
- 实验入口：`cd code/lab40-reward-model && uv sync && uv run run_lab40.py`


