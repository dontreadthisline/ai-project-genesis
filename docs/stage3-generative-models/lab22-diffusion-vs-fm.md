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

- 接口骨架：[`../../code/lab22-diffusion-vs-fm/skeleton.py`](../../code/lab22-diffusion-vs-fm/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab22-diffusion-vs-fm/selftest.py`](../../code/lab22-diffusion-vs-fm/selftest.py)
- 实验入口：`cd code/lab22-diffusion-vs-fm && uv sync && uv run run_lab22.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 方法 × 训练预算 $\{10k, 40k\}$ 步 | 同骨干 | loss、样本质量 |
| E2 | 方法 × NFE $\{2, 8, 32, 128\}$ | 训练后 | 质量-NFE 曲线 |
| E3 | DDPM（祖先）vs DDPM（DDIM）vs FM | 同 NFE | 随机性来源与质量 |
| E4 | 统一参数化重训：eps 参数化的 FM 目标 | — | 理论换算在实践中的差异 |

## 4. 预期图表

- 图 1：loss 曲线（注意两损失量级不同，分轴或归一化）。
- 图 2：质量-NFE 双曲线：FM 在少步区（NFE ≤ 8）通常占优；高 NFE 两端趋同。
- 图 3：E3 三列样本：祖先采样的多样性 vs ODE 确定性轨迹。
- 图 4：E4 对照：同一理论目标的两种参数化训练曲线差异（数值稳定性）。

## 5. 观测指标

`quality_vs_nfe`（两方法）、`final_losses`、`sample_grids`。

## 6. 结果分析

1. 用 1.1 的统一插值框架解释「两方法本质是同一族的不同调度」。
2. E2 的少步区间差异来自什么？（路径/调度的曲率；DDIM 是一阶积分直线路径的近似。）
3. 需要似然（密度估计、压缩）选谁？需要少步生成选谁？说明理由。

## 7. 验收清单

- [ ] T1-T4 通过；能写出统一插值 $a(t)x_1 + b(t)x_0$ 下两种目标的换算
- [ ] 完成 NFE-质量双曲线并解释交叉点
- [ ] 阶段验收汇总：能独立解释 VAE/GAN/Flow/DDPM/FM 五种范式的核心机制与取舍
