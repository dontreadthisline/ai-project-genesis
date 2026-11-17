---
title: "Lab 18：二维 Normalizing Flow"
lab: 18
stage: 3
---

对应大纲实验 18。目标数据：二维高斯混合（两个月亮亦可）。本实验目标是变量替换公式的直觉，不追求复杂架构。

## 0. 理论问题

1. 可逆变换下概率密度如何变换？Jacobian 行列式从哪来？
2. 什么样的变换既可逆又能高效算行列式？
3. 精确似然（对 VAE/GAN 的 ELBO/无密度）意味着什么？

## 1. 数学基础

### 1.1 变量替换公式（推导）

设 $z \sim p_z$，$x = f(z)$，$f$ 可微可逆。概率质量守恒：$z$ 空间小区域 $dz$ 的质量等于 $x$ 空间像区域 $dx$ 的质量：

$$
p_z(z)\, |dz| = p_x(x)\, |dx|, \qquad |dx| = \left| \det \frac{\partial f}{\partial z} \right| |dz|
$$

得 $p_x(x) = p_z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|$。对数似然：

$$
\log p_x(x) = \log p_z(z) - \log\left| \det \frac{\partial f}{\partial z} \right|, \qquad z = f^{-1}(x)
$$

行列式度量体积伸缩：变换在某方向压缩体积，密度必须增大补偿。训练 = 最大化数据点上的 $\log p_x$。

### 1.2 三角结构的行列式

任意可逆变换的 Jacobian 行列式是 $O(d^3)$（LU 分解）。上/下三角矩阵的行列式 = 对角线乘积，$O(d)$。RealNVP 仿射耦合层：把 $x$ 拆成两半，变换一半、条件于另一半：

$$
h_1 = x_1, \qquad h_2 = x_2 \odot \exp(s(x_1)) + t(x_1)
$$

Jacobian 是块三角：

$$
J = \begin{pmatrix} I & 0 \\ \partial h_2/\partial x_1 & \mathrm{diag}(\exp s) \end{pmatrix}
\ \Longrightarrow\ \det J = \exp\!\left( \sum_j s(x_1)_j \right)
$$

$s, t$ 是任意网络（不需可逆）：逆 $x_2 = (h_2 - t(h_1)) \odot \exp(-s(h_1))$ 显式可写。多层交错变换两半，积累表达能力。

### 1.3 采样与密度的对偶

采样：$z \sim \mathcal N(0, I) \to x = f(z)$（前向）。密度：$x \to z = f^{-1}(x)$，上式求值。flow 是唯一「采样便宜 + 密度精确」兼得的范式（VAE 密度只有下界，GAN 无密度）。

## 2. 实现任务

4-6 个耦合层（$s, t$ 为 2 层小 MLP），目标分布：两个高斯混合 $\tfrac12\mathcal N((-2,0), I) + \tfrac12\mathcal N((2,0), I)$。

```python
class AffineCoupling:        # 实现 forward / inverse / log_det
class Flow:                  # 堆叠 coupling，交替 mask 两半
    def sample(self, n): ...
    def log_prob(self, x): ...   # 1.1 公式累加 -log|det|
```

自测：

```python
# T1: 可逆性：random x, f^{-1}(f(x)) == x（rel err < 1e-5）
# T2: log_det：单层耦合的解析值 exp(Σs) 与数值 Jacobian slogdet 一致（2x2 数值可行）
# T3: 密度归一化：网格上数值积分 log_prob 的 exp ≈ 1（容差 2%）
# T4: 目标分布解析对照：训练后模型在两峰附近密度高、在 (0,0) 附近密度低
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 层数 $\{2, 4, 8\}$ | — | 负对数似然、双峰再现质量 |
| E2 | 训练前后 $z$-空间（数据经 $f^{-1}$）的形态 | — | 从混合高斯到标准正态的变形 |
| E3 | 逐步可视化：$f$ 的中间层把 $z$ 空间逐步弯折 | — | 密度传输动画帧 |
| E4 | 采样数 $\{100, 10^4\}$ 的直方图 vs 真实密度 | — | 密度估计与采样的一致性 |

## 4. 预期图表

- 图 1：三联图：$z$ 空间标准正态点云 / 训练后 $f(z)$ 点云 / 真实目标密度等高线。
- 图 2：数据经 $f^{-1}$ 回映射：训练前是分离双峰（在 z 空间仍是混合），训练后近似标准正则圆云。
- 图 3：负对数似然曲线收敛。
- 图 4：中间层形态：每层轻微弯折、总体组成大变形。

## 5. 观测指标

`final_nll`、`inv_consistency_err`、`density_integral`。

## 6. 结果分析

1. 用 1.1 的质量守恒解释：为什么变换在两峰之间的低谷区域必须有强压缩？
2. E1 中 2 层为何不足以拟合双峰（每层是剪切+缩放，需要多少弯折才能拆分质量）？
3. 对比 VAE：同为「$z$ 到 $x$ 的生成映射」，为什么 flow 能给精确似然而 VAE 只有 ELBO？

## 7. 验收清单

- [ ] T1-T4 通过；能白板推导变量替换公式与耦合层行列式
- [ ] 能解释三角 Jacobian 的 $O(d)$ 行列式为何是 flow 可扩展的关键
- [ ] 双峰目标被成功再现（可视化 + NLL 收敛）
