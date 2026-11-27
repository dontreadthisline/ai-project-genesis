---
title: "Lab 22：Diffusion 与 Flow Matching 对照"
lab: 22
stage: 3
---

对应大纲实验 22。依赖 Lab 19/21。同一数据（MNIST）、同一骨干（UNetLite）、同一预算下的系统对照。

## 0. 理论问题

1. DDPM 与 FM 的训练目标在数学上如何互相转换？
2. 两者的采样过程与步数瓶颈有何不同？
3. 什么场景选哪个？

## 1. 数学基础

### 1.1 目标的统一视角

DDPM 的 eps 预测：$x_t = \sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}\epsilon$，目标 $\epsilon = \frac{x_t - \sqrt{\bar\alpha_t}x_1}{\sqrt{1-\bar\alpha_t}}$。FM 直线路径：$x_t = (1-t)x_0 + tx_1$，目标 $x_1 - x_0$。

两者都是「噪声-数据插值 + 对插值求导」的特例：

$$
x_t = a(t)\, x_1 + b(t)\, x_0, \qquad u_t = a'(t)\, x_1 + b'(t)\, x_0
$$

DDPM 取 $a = \sqrt{\bar\alpha_t}, b = \sqrt{1-\bar\alpha_t}$（方差保持调度），FM 取 $a = t, b = 1-t$（线性调度）。目标间的换算：给定网络输出任一种参数化，另一个由 $x_1 = \frac{x_t - b(t)x_0}{a(t)}$ 反解。理论上仅差一个与 $(x_t, t)$ 相关的确定性重缩放。

### 1.2 采样与步数

DDPM 反向是马尔可夫链（祖先采样，每步加噪），其确定性版本 DDIM 对应 FM 的概率流 ODE 欧拉积分——同为 ODE 积分器。步数瓶颈来自 ODE 轨迹弯曲度：训练好的 FM 直线路径边际场接近直（尤其在 reflow 后），少步即可；DDPM 方差保持调度在高噪声端的场变化更剧烈。

### 1.3 对照维度清单

| 维度 | DDPM | FM（直线路径） |
|---|---|---|
| 训练目标 | 预测 $\epsilon$（噪声） | 预测 $x_1 - x_0$（速度） |
| 概率路径 | 方差保持 $\bar\alpha_t$ 调度 | 线性插值 |
| 超参 | $\beta_t$ 调度、$T$ | 路径形状（通常无） |
| 似然 | 有 ELBO（可算 NLL） | 无直接似然 |
| 采样 | 祖先链 / DDIM ODE | ODE（欧拉/Heun） |

## 2. 实现任务

共用 `UNetLite`（时间嵌入条件化同时服务两者）。实现统一训练循环，`method ∈ {ddpm, fm}` 切换损失与采样。

实现清单（签名与 TODO 见骨架源码）：

- `DDPM` — Lab 19 精简版
- `FM` — Lab 20/21 精简版
- `train_compare` — train_compare

- 接口骨架：[`../../code/lab22-diffusion-vs-fm/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab22-diffusion-vs-fm/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab22-diffusion-vs-fm/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab22-diffusion-vs-fm/selftest.py)
- 实验入口：`cd code/lab22-diffusion-vs-fm && uv sync && uv run run_lab22.py`


