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

### 1.2 KV cache 的可行性

因果注意力下位置 $t$ 的查询只与键值 $\le t$ 交互。生成第 $t+1$ 个 token 时，历史位置的 K、V 不变（各层一次算好缓存），只需：新 token 的 $q, k, v$（一次）、与缓存的注意力（一步）。无 cache 时每步重算全部 T 个位置的 K/V 及注意力矩阵——$O(T^2 d)$ 重复劳动；cache 后每步 $O(T d)$。代价：缓存占用 $2 \times L \times T \times d$ 内存（长上下文时的主要内存项）。

### 1.3 Prefill 与 decode

Prefill（处理 prompt）：全部位置并行，compute-bound（大矩阵乘）。Decode（逐 token）：batch=1 时是小 GEMV，memory-bound（权重带宽是瓶颈）——tokens/s 远低于 prefill 的并行吞吐。两阶段分离是推理服务的核心结构。

## 2. 实现任务

```python
def sample_token(logits, temperature, top_k, top_p, rng): ...
class KVCache:
    def append(self, k, v): ...    # (L, B, T, d) 累积
def generate_cached(model, prompt, max_new, cfg, cache): ...
def generate_naive(model, prompt, max_new, cfg): ...
```

自测：

```python
# T1: 采样等价性：cached 与 naive 生成在 greedy（T=0）下逐 token 一致
# # T2: 温度单调性：同一分布下采样熵随 T 单调增（MC 估计）
# # T3: top-p 自适应性：尖分布（熵 0.5）保留的候选数 < 平分布（熵 3）的
# # T4: 缓存正确性：手工前向一步的 K/V 与 cache 追加后一致
# # T5: 数值一致：T=1、top_k=V 时采样分布 == 原分布（频率检验）
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | 有无 KV cache | prompt 长度 $\{32, 128, 512\}$ | 每 token 延迟、加速比 |
| E2 | T $\in \{0.3, 0.7, 1.0, 1.5\}$ | top 关闭 | 生成多样性（distinct-n）vs 质量 |
| E3 | top-k / top-p 组合扫描 | — | 重复率 vs 多样性 |
| E4 | cache 内存 vs 上下文长度 | — | $2LTd$ 公式实测 |
| E5 | prefill vs decode 吞吐 | — | tokens/s 差距（量级） |

## 4. 预期图表

- 图 1：每 token 延迟 vs 生成位置：naive 随位置线性恶化（重算 $T^2$），cache 近乎平坦。加速比在 512 长度下达一个量级。
- 图 2：E2 的样本矩阵（T 扫描 × 同 prompt）：低温保守重复、高温混乱。
- 图 3：重复率-diversity 散点（各采样配置），标出 Pareto 前沿配置。
- 图 4：E5 吞吐柱状图（对数轴）：decode 的 tokens/s 比 prefill 低一到两个量级。

## 5. 观测指标

`latency_curve_cache_vs_naive`、`speedup_by_len`、`cache_mem_formula_check`、`diversity_by_T`。

## 6. 结果分析

1. 用 1.2 推导：prompt=512、生成 100 token，naive 与 cache 的注意力 FLOPs 比是多少？
2. E3 中 top-p 比 top-k 优势的场景？（分布形态随上下文变化时自适应。）
3. E5 的 memory-bound 现象对部署意味着什么？（batch 化、投机解码等方向的动机，只需定性。）

## 7. 验收清单

- [ ] T1-T5 通过；cache 与 naive 数值一致 + 加速实测
- [ ] 能推导温度重缩放后的分布形式与 cache 的复杂度
- [ ] 能解释 prefill/decode 的不对称
