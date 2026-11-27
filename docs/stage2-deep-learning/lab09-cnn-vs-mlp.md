---
title: "Lab 09：MLP 与 CNN 对照"
lab: 9
stage: 2
---

对应大纲实验 9。数据集：MNIST。

## 0. 理论问题

1. 卷积在数学上是什么运算？它对图像数据的哪些先验做了编码？
2. 参数共享和平移等变性分别带来什么？两者是什么关系？
3. 相同参数量级下，CNN 为什么在图像上胜过 MLP？

## 1. 数学基础

### 1.1 二维互相关（深度学习的「卷积」）

$$
y(i, j) = \sum_{u=1}^{K}\sum_{v=1}^{K} x(i+u-1,\ j+v-1)\, w(u, v) + b
$$

单通道单 kernel。输出空间尺寸（stride $s$、padding $p$、输入 $n$、kernel $K$）：

$$
m = \left\lfloor \frac{n + 2p - K}{s} \right\rfloor + 1
$$

多通道：输入 $C_{in}$ 通道时每个 kernel 是 $K \times K \times C_{in}$ 张量，$C_{out}$ 个 kernel 输出 $C_{out}$ 通道。参数量 $C_{out} \cdot K^2 \cdot C_{in} + C_{out}$，与输入空间尺寸无关。

### 1.2 平移等变性（证明）

设平移算子 $\tau$，卷积算子 $\mathcal K$。则 $\mathcal K(\tau x) = \tau(\mathcal K x)$：把 $(\tau x)(i,j) = x(i-1, j)$ 代入定义即得——求和窗口整体平移，权重的相对位置不变。卷积与平移可交换，故「特征出现在哪」不影响「是什么特征被检出」。全连接层不具备此性质：每个输入位置对应独立的权重列，平移后激活完全改变。

参数共享是等变性的实现手段：同一 kernel 在所有位置复用。池化在此基础上进一步给出（近似）平移不变性。

### 1.3 感受野

$L$ 层 $K\times K$ 卷积（stride 1）堆叠，第 $L$ 层每个单元的感受野：

$$
RF_L = 1 + L(K-1)
$$

归纳法：$RF_1 = K$；每加一层，感受野扩大 $K-1$。两层 $3\times3$（$RF=5$，参数 $18C^2$）与一层 $5\times5$（$RF=5$，参数 $25C^2$）等感受野，但前者更深、非线性更多、参数更少。

### 1.4 参数量对照

输入 $28\times28$：MLP 首层 $784 \times h$；CNN 首层 $32 \times (3^2 \cdot 1)$。等参数量配置下 MLP 缺少局部性与共享先验，需要更多数据才能学到等价结构——这正是实验 10 要量化的归纳偏置。

## 2. 实现任务

LeNet 风格：`conv5x5(s=2) 1→6 → tanh → avgpool2 → conv5x5(s=1) 6→16 → tanh → pool → flatten → fc120 → fc84 → fc10`。手写卷积前向（im2col 或直接双循环）与反向（前向的转置）。

实现清单（签名与 TODO 见骨架源码）：

- `Conv2d` — weight (C_out, C_in, K, K)
- `Pool2d` — 非重叠 max/avg，无参数
- `Flatten` — Flatten

- 接口骨架：[`../../code/lab09-cnn-vs-mlp/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab09-cnn-vs-mlp/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab09-cnn-vs-mlp/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab09-cnn-vs-mlp/selftest.py)
- 实验入口：`cd code/lab09-cnn-vs-mlp && uv sync && uv run run_lab09.py`


