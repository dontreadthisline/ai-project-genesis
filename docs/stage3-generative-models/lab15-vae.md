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

<iframe src="resources/figures/vae-latent.html" style="width:100%;height:404px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="VAE 的二维潜空间"></iframe>

<p align="center"><b>图 15.1</b>：潜空间簇 vs 标准正态先验</p>

### 1.3 两项的闭式形式

KL 项（Lab 04 §1.5 高斯对高斯，先验标准正态）：

$$
D_{\mathrm{KL}} = \tfrac12 \sum_{j=1}^{d}\left( \mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2 \right)
$$

重构项：二值 MNIST 用 Bernoulli 解码 $\log p(x|z) = \sum x\log\hat x + (1-x)\log(1-\hat x)$；或高斯解码退化为 MSE。重构项期望用单样本估计（Lab 03）。

<iframe src="resources/figures/elbo-tradeoff.html" style="width:100%;height:374px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="重构与 KL 的拉锯"></iframe>

<p align="center"><b>图 15.2</b>：两项的此消彼长</p>

### 1.4 重参数化与 $\beta$

$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon,\ \epsilon \sim \mathcal N(0, I)$：随机性外移到 $\epsilon$，$\nabla_{\phi}$ 穿过确定性路径（Lab 01 Cholesky 采样同构，Lab 16 专门对比）。$\beta$-VAE：$\mathcal L = \text{重构} - \beta\,\mathrm{KL}$，$\beta > 1$ 换取更解耦的潜空间、付出重构模糊的代价。

## 2. 实现任务

encoder：$784 \to 256 \to (\mu, \log\sigma^2)$ 各 2 维；decoder：$2 \to 256 \to 784$ sigmoid。

实现清单（签名与 TODO 见骨架源码）：

- `Encoder` — 输出 mu (B,2), logvar (B,2)
- `Decoder` — 返回 (loss, recon_term, kl_term)
- `reparameterize` — 返回 (loss, recon_term, kl_term)
- `vae_loss` — 返回 (loss, recon_term, kl_term)

- 接口骨架：[`../../code/lab15-vae/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab15-vae/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab15-vae/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab15-vae/selftest.py)
- 实验入口：`cd code/lab15-vae && uv sync && uv run run_lab15.py`


