---
title: "Lab 21：MNIST Flow Matching"
lab: 21
stage: 3
---

对应大纲实验 21。数据集：MNIST。依赖 Lab 19 的网络骨架、Lab 20 的损失与采样。

## 0. 理论问题

1. 图像域 FM 与二维 FM 的差别只在网络容量与维度吗？
2. 采样步数与样本质量的权衡如何定量刻画？
3. 时间嵌入在网络中起什么作用？

## 1. 数学基础

### 1.1 图像域的直线路径

$x_0 \sim \mathcal N(0, I_{784})$，$x_1 = $ 归一化到 $[-1,1]$ 的图像，$x_t = (1-t)x_0 + tx_1$，目标恒速场 $u = x_1 - x_0$，损失与 Lab 20 §1.3 完全相同。维度从 2 到 784：理论上无新公式，实践上要求 $v_\theta$ 有图像归纳偏置（复用 Lab 09 的卷积骨干 + Lab 11 残差块）。

### 1.2 时间条件化

$v_\theta(x, t)$ 必须对 $t$ 条件化：不同时刻的最优速度不同（边际场的时变性）。正弦嵌入 $t \mapsto [\sin 2\pi f_k t, \cos 2\pi f_k t]$ 后以加法或 FiLM（$\gamma(t) \odot h + \beta(t)$）注入每块。与 DDPM 的 $\epsilon_\theta(x_t, t)$ 完全同构：一个是噪声水平、一个是路径位置。

<iframe src="resources/figures/time-embedding.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="时间条件化"></iframe>

<p align="center"><b>图 21.1</b>：v(x, t) 随 t 改变</p>

### 1.3 NFE-质量权衡

函数评估次数（NFE）= 积分步数。欧拉步误差 $O(\Delta t)$ 全局 $O(\Delta t)$：步数减半误差近似翻倍。图像质量度量（本课程用感知检查 + 简单像素统计；FID 可选）随 NFE 的改善曲线是采样成本的核心指标，与 Lab 36（KV cache 加速解码）、Lab 19（扩散步数）同属「计算-质量前沿」主题。

<iframe src="resources/figures/nfe-quality.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="NFE-质量前沿"></iframe>

<p align="center"><b>图 21.2</b>：质量 vs 积分步数</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `TimeEmbedding` — sinusoidal -> MLP
- `ResBlockT` — ResBlock + 时间 FiLM
- `UNetLite` — 下采样-上采样 + 跳跃，输入 (x, t)
- `cfm_loss` — 同 Lab 20，域换成图像
- `sample` — 欧拉积分 + 最后一步 clamp 到 [-1,1]

- 接口骨架：[`../../code/lab21-flow-matching-mnist/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab21-flow-matching-mnist/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab21-flow-matching-mnist/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab21-flow-matching-mnist/selftest.py)
- 实验入口：`cd code/lab21-flow-matching-mnist && uv sync && uv run run_lab21.py`


