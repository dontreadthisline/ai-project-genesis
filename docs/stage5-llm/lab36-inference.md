---
title: "Lab 36：推理优化"
lab: 36
stage: 5
---

对应大纲实验 36。依赖 Lab 35 的训练模型。

## 0. 理论问题

1. 温度、top-k、top-p 分别在修改分布的什么？
2. KV cache 为什么可行？省掉的到底是什么计算？
3. Prefill 与 decode 两阶段的计算特性有何本质不同？

## 1. 数学基础

### 1.1 采样策略

$$
p_T(x) = \frac{p(x)^{1/T}}{\sum_{x'} p(x')^{1/T}}
$$

温度 $T$：logits 整体除以 $T$。$T \to 0$ 退化为 argmax（贪心），$T \to \infty$ 趋于均匀。$T$ 保持分布的**秩**不变，只改熵（Lab 04：KL(均匀‖采样分布) 随 T 单调）。Top-k：保留前 k 个、其余置 0 后重归一化（硬截断，截断量固定）。Top-p（nucleus）：保留累计概率达 p 的最小集合（自适应截断：分布尖时少保留、分布平时多保留）。

<iframe src="resources/figures/temperature-shapes.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="温度与截断的分布形态"></iframe>

<p align="center"><b>图 36.2</b>：同分布不同 T</p>

### 1.2 KV cache 的可行性

因果注意力下位置 $t$ 的查询只与键值 $\le t$ 交互。生成第 $t+1$ 个 token 时，历史位置的 K、V 不变（各层一次算好缓存），只需：新 token 的 $q, k, v$（一次）、与缓存的注意力（一步）。无 cache 时每步重算全部 T 个位置的 K/V 及注意力矩阵——$O(T^2 d)$ 重复劳动；cache 后每步 $O(T d)$。代价：缓存占用 $2 \times L \times T \times d$ 内存（长上下文时的主要内存项）。

<iframe src="resources/figures/kv-cache.html" style="width:100%;height:394px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="KV Cache：省掉重复劳动"></iframe>

<p align="center"><b>图 36.1</b>：无缓存（全重算）vs 缓存（增量）</p>

### 1.3 Prefill 与 decode

Prefill（处理 prompt）：全部位置并行，compute-bound（大矩阵乘）。Decode（逐 token）：batch=1 时是小 GEMV，memory-bound（权重带宽是瓶颈）——tokens/s 远低于 prefill 的并行吞吐。两阶段分离是推理服务的核心结构。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `sample_token` — sample_token
- `KVCache` — (L, B, T, d) 累积
- `generate_cached` — generate_cached
- `generate_naive` — generate_naive

- 接口骨架：[`../../code/lab36-inference/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab36-inference/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab36-inference/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab36-inference/selftest.py)
- 实验入口：`cd code/lab36-inference && uv sync && uv run run_lab36.py`


