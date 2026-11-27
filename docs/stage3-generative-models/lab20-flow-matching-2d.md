---
title: "Lab 20：二维 Flow Matching"
lab: 20
stage: 3
---

对应大纲实验 20。目标数据：二维（两个月亮或双峰）。依赖 Lab 03（期望估计）、Lab 18（密度直觉）。

## 0. 理论问题

1. 概率路径、向量场、ODE 三者什么关系？为什么「速度场积分 = 分布演化」？
2. 条件流匹配为什么能用回归目标学出边际向量场？
3. 直线路径（Rectified Flow）的特别之处？

## 1. 数学基础

### 1.1 连续性方程

分布 $\rho_t$ 沿向量场 $v_t$ 演化需满足质量守恒：

$$
\frac{\partial \rho_t}{\partial t} + \nabla \cdot (\rho_t v_t) = 0
$$

若 $x(t)$ 服从 ODE $\dot x = v_t(x)$、初始 $x(0) \sim \rho_0 = \mathcal N(0, I)$，则 $x(t)$ 的分布就是由该方程传输的 $\rho_t$。生成 = 积分 ODE 从 $\rho_0$ 到 $\rho_1 \approx p_{data}$。与 DDPM 的联系：每个扩散过程都有对应的概率流 ODE（漂移改写），FM 直接学这个 ODE 的右端。

### 1.2 条件流匹配（推导骨架）

直接回归边际场 $v_t$ 不可行（它由边际 $\rho_t$ 定义，而 $\rho_t$ 未知）。构造条件概率路径 $\rho_t(x | x_1)$：以某个数据点 $x_1$ 为目标的简单路径，其条件速度场 $u_t(x | x_1)$ 已知。关键恒等式（边缘化）：

$$
\rho_t(x) = \int \rho_t(x|x_1)\, p_{data}(x_1)\, dx_1 \ \Longrightarrow\ v_t(x) = \mathbb E_{x_1 \sim p_{data},\, x \sim \rho_t(\cdot|x_1)}\left[ u_t(x | x_1) \mid x \right]
$$

回归目标的期望版（对学习足够的版本）：

$$
\mathcal L_{CFM} = \mathbb E_{t,\, x_1 \sim p_{data},\, x_0 \sim \mathcal N(0,I)} \left\| v_\theta(t, x_t) - u_t(x_t | x_0, x_1) \right\|^2
$$

最小化它得到的 $v_\theta$ 与边际场在分布意义下一致。样本 $(x_0, x_1, t)$ 全部显式可采——无 ELBO、无对抗、无噪声调度超参（路径本身是选择）。

### 1.3 直线路径（Rectified Flow）

最简单的路径：$x_t = (1-t)x_0 + t x_1$，条件速度场恒定 $u_t = x_1 - x_0$（对 $t$ 微分即得）。损失：

$$
\mathcal L = \mathbb E\left\| v_\theta(t, x_t) - (x_1 - x_0) \right\|^2
$$

边际场 $v_t$ 是「指向数据的直线路径平均」：在高密度重叠区，多个目标的平均方向起主导——这就是弯曲的来源。采样：欧拉法 $x_{t+\Delta} = x_t + \Delta\, v_\theta(t, x_t)$。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `VelocityNet` — (t, x) -> v，t 用 sinusoidal 嵌入
- `cfm_loss` — 1.3 直线路径损失
- `euler_sample` — euler_sample

- 接口骨架：[`../../code/lab20-flow-matching-2d/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab20-flow-matching-2d/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab20-flow-matching-2d/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab20-flow-matching-2d/selftest.py)
- 实验入口：`cd code/lab20-flow-matching-2d && uv sync && uv run run_lab20.py`


