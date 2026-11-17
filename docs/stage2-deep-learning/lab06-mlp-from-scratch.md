---
title: "Lab 06：从零实现 MLP"
lab: 6
stage: 2
---

对应大纲实验 6。数据集：MNIST。产出：`run_lab06.py`。

## 0. 理论问题

1. 前向传播和反向传播在计算上分别做了什么？为什么反向传播比数值微分高效？
2. 交叉熵加 softmax 的梯度为什么有如此简单的形式？
3. 参数初始化的方差为什么影响训练成败？

## 1. 数学基础

### 1.1 线性层与链式法则

线性层 $y = Wx + b$（$W \in \mathbb R^{out \times in}$）。对输入的 Jacobian 是 $W$，对参数的梯度由链式法则给出。反向传播的核心对象是 VJP（向量-Jacobian 积）：

$$
\bar x = W^T \bar y, \qquad \bar W = \bar y\, x^T, \qquad \bar b = \bar y
$$

其中 $\bar{(\cdot)}$ 表示上游梯度。整网反向 = 逆序逐层应用 VJP，代价与前向同阶；数值微分对 $P$ 个参数各需两次前向，代价 $O(P)$ 倍。

### 1.2 Softmax 与交叉熵的合并梯度

softmax：$p_i = e^{z_i} / \sum_j e^{z_j}$，交叉熵 $L = -\log p_y$。推导 $\partial L / \partial z$：

$$
\frac{\partial L}{\partial z_i} = -\frac{\partial \log p_y}{\partial z_i}, \qquad
\frac{\partial p_j}{\partial z_i} = p_j(\delta_{ij} - p_i)
$$

当 $i = y$ 时 $\partial \log p_y/\partial z_y = \frac{1}{p_y} p_y(1-p_y) = 1 - p_y$；当 $i \neq y$ 时 $= -p_i$。合并：

$$
\boxed{\ \frac{\partial L}{\partial z} = p - \mathbf 1_y\ }
$$

梯度就是「预测概率减独热标签」。这是 Lab 04「CE 是分类标准损失」在梯度层面的原因：错误越大概率越大，梯度无饱和区。

### 1.3 数值梯度检验

中心差分 $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$ 误差 $O(h^2)$（泰勒展开相消）。取 $h = 10^{-6}$（float64），解析梯度与数值梯度的相对误差

$$
\mathrm{err} = \frac{\|\nabla_a - \nabla_n\|_2}{\|\nabla_a\|_2 + \|\nabla_n\|_2 + \epsilon}
$$

应小于 $10^{-6}$。这是所有手写反向传播的标准验证手段。

### 1.4 初始化

若线性层输入各分量独立零均值方差 $\sigma_x^2$，则 $y_i = \sum_j W_{ij} x_j$ 的方差为 $in \cdot \mathrm{Var}(W) \cdot \sigma_x^2$。要保持逐层方差稳定（前向不爆炸不消失）需 $\mathrm{Var}(W) = 2/(fan_{in} + fan_{out})$（Xavier，配 tanh）；配合 ReLU 时一半神经元死亡，改用 $\mathrm{Var}(W) = 2/fan_{in}$（He 初始化）。

## 2. 实现任务

结构 $784 \to 256 \to 128 \to 10$，激活 ReLU，输出交叉熵。**用 MLX 实现**（`mx.array` 张量运算），但**不用框架自动微分**：手写 forward/backward，不调用 `mx.grad`。数值梯度检验用 `dtype=mx.float64`（MLX 的 fp64 在 CPU 流水线执行，精度满足 $10^{-6}$ 相对误差）。

```python
class Linear:
    def __init__(self, fan_in, fan_out): ...   # He 初始化 W, 零初始化 b
    def forward(self, x): ...                       # 缓存 x
    def backward(self, grad_out): ...               # 返回 grad_in，累积 self.dW/db

class MLP:
    def __init__(self, sizes): ...             # 逐层 Linear + ReLU
    def forward(self, x): ...
    def backward(self, grad): ...                   # 逆序 VJP
    def params_grads(self): ...                     # [(W, b, dW, db), ...]

def softmax_cross_entropy(logits, y_onehot):
    """数值稳定：logits 减最大值。返回 (loss, dlogits)。dlogits 用 1.2 的闭式解。"""
```

自测：

```python
# T1: softmax-CE 梯度与数值梯度一致（float64，rel err < 1e-6，随机 20 组）
# T2: 全网络 backward 与数值梯度一致（逐参数抽样检查，rel err < 1e-6）
# T3: He 初始化后各层激活方差在 [0.5, 2] 区间（随机 512 个样本过网络）
# T4: softmax 数值稳定：logits = [1000, 1000.1, 999.9] 不产生 nan/inf
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 手写梯度 vs 数值梯度 | — | 相对误差分布 |
| E2 | 初始化：He / 全零 / 标准差 1.0 高斯 | 其余固定 | 训练 loss 是否下降 |
| E3 | 隐藏层：无 / 1 层 / 2 层 | 参数量近似对齐 | 测试准确率、收敛速度 |
| E4 | 训练前 / 中 / 后各层梯度范数与激活方差 | — | 是否存在消失或爆炸层 |

## 4. 预期图表

- 图 1：训练/验证 loss 与 accuracy 曲线（约 10 epoch，测试准确率 97%-98%）。
- 图 2：E2 四条 loss 曲线：全零初始化完全不动（对称性无法打破），大方差高斯发散或震荡，He 正常下降。
- 图 3：逐层梯度范数柱状图（反向第 1 层梯度最小但与首层同数量级）。

## 5. 观测指标（metrics.json）

`final_test_acc`、`per_layer_grad_norm`、`per_layer_activation_var`、`E2_final_loss_by_init`。

## 6. 结果分析

1. E2 中全零初始化为何不动？（提示：梯度对第二层为零，对称性。）
2. 观察 loss 从随机预测的 $\ln 10 \approx 2.303$ 起步，第一个 epoch 内快速下降到多少？这一段主要由哪几层的学习贡献？
3. 用 1.2 的闭式解解释：为什么 CE 训练早期的梯度规模与「预测错误程度」成正比。

## 7. 验收清单

- [ ] T1-T4 通过，未使用框架自动微分
- [ ] 能白板推导 softmax-CE 梯度 $p - \mathbf 1_y$ 与 VJP 公式
- [ ] 能解释 He 初始化方差的来源与 ReLU 的关系
- [ ] MNIST 测试准确率 $\geq 97\%$
