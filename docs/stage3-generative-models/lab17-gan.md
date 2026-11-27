---
title: "Lab 17：实现基础 GAN"
lab: 17
stage: 3
---

对应大纲实验 17。数据集：MNIST。依赖 Lab 04（JS = 和的 KL 平均）、Lab 06。

## 0. 理论问题

1. 固定 G 时最优判别器是什么？把最优 D 代入后 G 的目标变成什么？
2. 极小极大博弈为什么天然不稳定？两个网络互相追逐意味着什么？
3. Mode collapse 的机制是什么？如何观测？

## 1. 数学基础

### 1.1 目标函数

$$
\min_G \max_D V(D, G), \qquad
V = \mathbb E_{x \sim p_{data}}[\log D(x)] + \mathbb E_{z \sim p(z)}[\log(1 - D(G(z)))]
$$

### 1.2 最优判别器（推导）

固定 G，对单个 $x$，$D$ 的最优值最大化 $p_{data}(x)\log D + p_g(x)\log(1-D)$。对 $D$ 求导置零：

$$
\frac{p_{data}}{D} - \frac{p_g}{1-D} = 0 \ \Longrightarrow\ D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

### 1.3 全局最优：JS 散度（推导）

把 $D^*$ 代回 $V$。记 $a = p_{data}, b = p_g$：

$$
V(G, D^*) = a\log\frac{a}{a+b} + b\log\frac{b}{a+b} + (a + b)\log 2 - (a+b)\log 2 \cdot 1
$$

逐点整理（注意两个期望都是对同一 $x$ 空间积分）：

$$
C(G) = V(D^*, G) = -\log 4 + \mathrm{KL}\left(p_{data} \,\|\, \tfrac{p_{data}+p_g}{2}\right) + \mathrm{KL}\left(p_g \,\|\, \tfrac{p_{data}+p_g}{2}\right) = -\log 4 + 2\,\mathrm{JS}(p_{data} \| p_g)
$$

最小值 $-\log 4$ 当且仅当 $p_g = p_{data}$：博弈的 Nash 均衡即分布匹配。

### 1.4 实践形态与不稳定

原始 $\log(1-D(G(z)))$ 在 G 差时梯度饱和，实践改为最大化 $\log D(G(z))$（非饱和损失）——它不再对应同一 JS 理论。JS 梯度消失问题：不重叠的低维流形间 JS 为常数 $\log 2$，梯度为零，训练信号断裂。Mode collapse：G 发现个别 $z$ 区域的样本能稳定骗过 D，就把多种 $z$ 都映射到同一输出模式——损失看不出异常，只能靠生成多样性观测。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `Generator` — z(64) -> 256 -> 512 -> 784 sigmoid
- `Discriminator` — 784 -> 512 -> 256 -> 1 sigmoid
- `d_loss` — BCE with logits
- `g_loss` — g_loss

- 接口骨架：[`../../code/lab17-gan/skeleton.py`](../../code/lab17-gan/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab17-gan/selftest.py`](../../code/lab17-gan/selftest.py)
- 实验入口：`cd code/lab17-gan && uv sync && uv run run_lab17.py`


## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | D/G 更新比 $\{1:1, 5:1\}$ | — | D 准确率轨迹、样本质量 |
| E2 | G/D 学习率错配（G 慢 10 倍等） | — | 崩溃模式 |
| E3 | 训练阶段快照（1/5/20/50 epoch） | — | 样本演化、mode collapse 出现 |
| E4 | 固定噪声种子可视化同一批 z 的输出漂移 | — | collapse 的直接观测 |

## 4. 预期图表

- 图 1：D/G loss 双曲线：任何一条 loss 趋 0 都意味着失衡（D 完胜或 G 骗过一切），理想区间是围绕某个非零值的拉锯。
- 图 2：D 准确率曲线：健康训练在 60%-90% 震荡；趋 100% 说明 D 太强。
- 图 3：四阶段样本网格：从噪声块到数字形态；记录 collapse 时同类数字铺满网格。
- 图 4：E4 同 z 漂移：collapse 模型下不同 z 映射到几乎相同输出。

## 5. 观测指标

`d_acc_curve`、`g_loss_curve`、`sample_grids`、`mode_count`（用简单聚类或类别分类器估计生成多样性）。

## 6. 结果分析

1. 用 1.3 说明「训练成功时 JS 最小」，再解释为什么实践中 loss 数值并不对应 JS（非饱和替换、有限样本）。
2. E1 中 5:1 的深 D 训练是否带来更好样本？D 过强的证据是什么？
3. 与 Lab 15 对照：VAE 的潜空间有先验结构约束、GAN 无显式密度——两者的失败模式（模糊 vs collapse）与这一点有何对应？

## 7. 验收清单

- [ ] T1-T4 通过；能白板推导 $D^*$ 与 $C(G) = -\log4 + 2\,\mathrm{JS}$
- [ ] 能从 loss/acc 曲线诊断训练失衡
- [ ] 观测到至少一次 mode collapse 并给出解释
