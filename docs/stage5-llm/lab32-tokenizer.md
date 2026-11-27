---
title: "Lab 32：Tokenizer 实验"
lab: 32
stage: 5
---

对应大纲实验 32。语料：TinyStories 子集（或内置短语文本）。

## 0. 理论问题

1. 字符/词/子词三级方案的权衡是什么？
2. BPE 的合并顺序为什么由频率驱动？
3. 词表大小如何影响序列长度、embedding 参数量与下游学习难度？

## 1. 数学基础

### 1.1 分词的目标

语言模型在 token 序列上定义 $p(x_1, \dots, x_T) = \prod p(x_t | x_{<t})$。分词器决定「文本 → 整数序列」的映射 $\mathcal T$。设计目标的三方拉扯：

- **序列长度**：token 越长序列越短，注意力/循环的计算 $\propto T^2$ 或 $T$ 都受益；
- **词表大小**：$|V|$ 决定 embedding 矩阵 $|V| \times d$ 与 softmax 维度；
- **组合性**：子词保证任意新字符串可编码（无 OOV），词级做不到。

字符级（$|V| \approx 100$，无 OOV，序列最长）与词级（$|V| \sim 10^5$，序列短，OOV 致命）是两个极端，BPE 居中。

### 1.2 BPE 训练算法

初始词表 = 字符集。重复直到词表达到 $|V|$：统计当前所有「相邻 token 对」频率，把最高频对合并为新 token。贪心：每一步最大化压缩率。编码：按训练得到的合并表依序应用同样的合并。

关键性质：**合并是确定性序列**——编码时必须按训练时的优先级顺序应用合并，乱序会得到不同切分。

<iframe src="resources/figures/bpe-merge.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="BPE：从字符到子词"></iframe>

<p align="center"><b>图 32.1</b>：低, 最低, 低频 → 合并成块</p>

### 1.3 压缩率

$$
\text{压缩率} = \frac{\text{字符数}}{\text{token 数}}
$$

随合并数增加单调不减。理论极限：语料的经验熵 $H$（Lab 04）给出平均码长下界（token 的 $\log_2 |V|$ 比特/token），BPE 是贪心近似而非最优编码。

<iframe src="resources/figures/compression-curve.html" style="width:100%;height:364px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="词表大小的边际收益"></iframe>

<p align="center"><b>图 32.2</b>：压缩率 vs |V|</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `train_bpe` — 返回 merges（有序合并表）与 vocab
- `bpe_encode` — bpe_encode
- `bpe_decode` — bpe_decode
- `char_encode` — char_encode

- 接口骨架：[`../../code/lab32-tokenizer/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab32-tokenizer/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab32-tokenizer/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab32-tokenizer/selftest.py)
- 实验入口：`cd code/lab32-tokenizer && uv sync && uv run run_lab32.py`


