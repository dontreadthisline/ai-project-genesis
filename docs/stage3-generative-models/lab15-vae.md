---
title: "Lab 15：实现 VAE"
lab: 15
stage: 3
---

对应大纲实验 15。数据集：MNIST。依赖 Lab 02（高斯条件）、Lab 04（KL 闭式解）、Lab 06 训练框架。

## 0. 理论问题

1. 为什么不能直接最大化 $\log p(x)$？ELBO 是怎么绕过这个困难的？
2. KL 项与重构项如何互相牵制？$\beta$ 权重改变什么？
3. 二维潜空间的形态说明了什么？

## 1. 数学基础

### 1.1 边际似然的不可解性

潜变量模型 $p_\theta(x) = \int p_\theta(x|z) p(z)\, dz$，$z \sim \mathcal N(0, I)$。对连续高维 $z$ 该积分不可解析；$\log p_\theta(x) = \log \int \cdots$ 中 log 与积不可交换，直接 Monte Carlo 估计的是 $\log$ 的下界之外（ Jensen 给出的是方向）。

### 1.2 ELBO 推导（Jensen 不等式，完整）

引入任意 $q_\phi(z|x)$：

$$
\log p_\theta(x) = \log \int p_\theta(x, z)\, dz
= \log \mathbb E_{q_\phi(z|x)}\left[ \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]
\ge \mathbb E_{q_\phi(z|x)}\left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]
$$

（log 凹，Jensen。）右边即 ELBO，拆开：

$$
\mathrm{ELBO} = \underbrace{\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重构项}} - \underbrace{D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))}_{\text{KL 正则项}}
$$

差值恰是 $\log p_\theta(x) - \mathrm{ELBO} = D_{\mathrm{KL}}(q_\phi(z|x) \| p_\theta(z|x))$：近似后验越接近真后验，界越紧。

### 1.3 两项的闭式形式

KL 项（Lab 04 §1.5 高斯对高斯，先验标准正态）：

$$
D_{\mathrm{KL}} = \tfrac12 \sum_{j=1}^{d}\left( \mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2 \right)
$$

重构项：二值 MNIST 用 Bernoulli 解码 $\log p(x|z) = \sum x\log\hat x + (1-x)\log(1-\hat x)$；或高斯解码退化为 MSE。重构项期望用单样本估计（Lab 03）。

### 1.4 重参数化与 $\beta$

$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon,\ \epsilon \sim \mathcal N(0, I)$：随机性外移到 $\epsilon$，$\nabla_{\phi}$ 穿过确定性路径（Lab 01 Cholesky 采样同构，Lab 16 专门对比）。$\beta$-VAE：$\mathcal L = \text{重构} - \beta\,\mathrm{KL}$，$\beta > 1$ 换取更解耦的潜空间、付出重构模糊的代价。

## 2. 实现任务

encoder：$784 \to 256 \to (\mu, \log\sigma^2)$ 各 2 维；decoder：$2 \to 256 \to 784$ sigmoid。

```python
class Encoder: ...    # 输出 mu (B,2), logvar (B,2)
class Decoder: ...
def reparameterize(mu, logvar, rng): ...
def vae_loss(x, recon, mu, logvar, beta=1.0): ...   # 返回 (loss, recon_term, kl_term)
```

自测：

```python
# T1: KL 闭式对照：随机制 (mu, logvar) 手算 1.3 公式 == vae_loss 的 kl 项（1e-6）
# T2: logvar 数值稳定：encoder 输出 clamp 到 [-10, 10]，exp 不上溢
# T3: ELBO ≤ log p 的方向检查：训练后 ELBO 单调上升
# T4: 重参数化梯度存在：对 mu 的数值梯度 == 解析梯度（单样本噪声固定时，rel err < 1e-5）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | $\beta \in \{0.1, 0.5, 1, 4\}$ | 训练 20 epoch | 潜空间图、重构 MSE、KL 值 |
| E2 | epoch 扫描 | — | 重构/KL 两项的此消彼长 |
| E3 | 潜变量插值（网格 $[-3,3]^2$ 解码） | 训练后模型 | 数字形态的连续过渡 |
| E4 | 潜维 $\{2, 8, 32\}$ | — | 重构质量与 KL/样本组织的差异 |

## 4. 预期图表

- 图 1：二维潜空间散点按数字类别着色：$\beta=1$ 时簇按数字分离且大致填满标准正态区域；$\beta$ 小则远离先验（KL 大、簇散乱），$\beta$ 大则向原点塌缩（KL → 0、重构糊）。
- 图 2：重构/KL 双轴曲线：训练初期 KL 先被压低，重构随后上升。
- 图 3：$20\times20$ 解码网格：一维方向渐变（笔画粗细/数字类别），簇间过渡带模糊。
- 图 4：E4 重构对比：潜维增大重构更清晰，但 2 维已足以保留主要流形结构。

## 5. 观测指标

`recon_mse`、`kl_by_beta`、`latent_means_norm`、`interp_grid`。

## 6. 结果分析

1. E1 中 KL 值随 $\beta$ 的变化曲线说明两项在争夺什么容量？
2. 图 3 的过渡带为什么模糊？（先验区域 $p(z)$ 中间地带解码器未见足够数据。）
3. 用 1.2 的紧度公式解释：训练好后 $q_\phi(z|x)$ 是否接近真实后验？（观察二维后验椭圆是否单峰紧凑。）

## 7. 验收清单

- [ ] T1-T4 通过；能白板完成 ELBO 的 Jensen 推导与 KL 闭式解
- [ ] 能解释重构-KL 的权衡与 $\beta$ 的作用
- [ ] 潜空间图呈现按类分离的连续结构
