---
title: "Lab 11：Residual Connection"
lab: 11
stage: 2
---

对应大纲实验 11。依赖 Lab 09。

## 0. 理论问题

1. 深度增加时，为什么普通网络的梯度会退化？
2. 残差连接在反向传播中改变了什么？
3. 残差网络的「恒等映射」思想为什么对任意深度有效？

## 1. 数学基础

### 1.1 普通深网的梯度衰减

$L$ 层复合 $h_L = f_L \circ \cdots \circ f_1(h_0)$，链式法则：

$$
\frac{\partial h_L}{\partial h_0} = \prod_{l=1}^{L} J_l, \qquad
\left\| \frac{\partial h_L}{\partial h_0} \right\| \le \prod_l \|J_l\| \le \left( \max_l \|J_l\| \right)^L
$$

若每层 Jacobian 谱范数略小于 1（例如 0.9），20 层后梯度 $\sim 0.9^{20} \approx 0.12$，50 层后 $\sim 0.005$：浅层梯度消失，优化停滞。若略大于 1 则指数爆炸。乘性结构是根源。

### 1.2 残差块的梯度高速公路

残差块 $h_{l+1} = h_l + F(h_l)$：

$$
\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F}{\partial h_l}
$$

跨 $L$ 个残差块：

$$
\frac{\partial h_L}{\partial h_0} = \prod_l \left( I + J_{F,l} \right) = I + \sum_l J_{F,l} + \sum_{l<m} J_{F,m} J_{F,l} + \cdots
$$

展开式中恒等项 $I$ 永远存在：任何一条「纯恒等路径」的梯度贡献为 1，不随深度衰减；$F$ 的小扰动项只做加性修正。梯度下界不再乘性缩小——这就是 highway 的含义：即使 $F$ 学不到东西，梯度也能无衰减地穿过。

### 1.3 与初始化的配合

若 $F$ 最后一层初始化为零（常见做法），初始网络是严格恒等映射 $h_L = h_0$，训练从「什么都不做」出发逐步增加残差修正。对比普通网络：再好的初始化也无法让 50 层复合恒等于恒等（误差逐层累积）。

## 2. 实现任务

```python
class PlainConvBlock:   # conv-bn-relu-conv-bn + (可选下采样)
class ResidualBlock:    # 同上 + 跳跃连接；下采样时 1x1 conv 投影对齐维度
def make_plain_cnn(depth): ...
def make_res_cnn(depth): ...   # 参数量与 plain 对齐（±15%）
```

自测：

```python
# T1: ResidualBlock 在 F=0（零初始化最后 conv）时输出 == 输入
# T2: 随机权重下，反向传播到第 1 层的梯度范数：res 网络随深度下降缓慢、
#     plain 网络显著更小（深度 20 对照，比值 > 5 倍）
# T3: 深度增加时前向激活范数：res 网络各层范数稳定，plain 网络可能衰减
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 深度 $\{4, 8, 16, 32\}$ × plain/res | 参数量对齐 | test acc、首层梯度范数 |
| E2 | 训练初期（前 3 epoch）loss 能否下降 | 深度 32 | 深网可训练性 |
| E3 | F 零初始化 vs 标准初始化（res） | — | 初始损失与收敛速度 |
| E4 | 去掉 BN 的 res vs plain（深度 16） | — | 两者对 norm 的依赖 |

## 4. 预期图表

- 图 1：acc vs 深度：plain 在深度 16 后退化（甚至不如浅层），res 持平或略升。
- 图 2：首层梯度范数 vs 深度（对数 y 轴）：plain 近似指数下降，res 平缓。
- 图 3：深度 32 的 loss 曲线：res 正常下降，plain 停滞在高 loss。
- 图 4：E3 两条曲线：零初始化起始 loss 更低（恒等映射）。

## 5. 观测指标

`acc_by_depth`、`first_layer_grad_norm_by_depth`、`trainable_flag`（3 epoch 内 loss 下降 > 10%）。

## 6. 结果分析

1. 用 1.2 的展开式解释图 2 中 res 的梯度下界。
2. E1 中 res 在深度 4 时是否优于 plain？（通常差距小——收益随深度才显现。）
3. 恒等项 $I$ 的存在是否意味着 res 网络等价于浅网？（否：$F$ 项学到的是对恒等的修正，深层复合仍增强表达能力。）

## 7. 验收清单

- [ ] T1-T3 通过；能白板推导 $I + J_F$ 的展开与梯度高速公路论证
- [ ] 复现 plain 深网退化、res 不退化的深度曲线
- [ ] 能解释零初始化残差末端与恒等映射初始状态的关系
