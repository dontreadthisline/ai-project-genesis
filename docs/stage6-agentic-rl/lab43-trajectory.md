---
title: "Lab 43：Agent Trajectory"
lab: 43
stage: 6
---

对应大纲实验 43。依赖 Lab 42。把「跑一次任务」变成「可统计、可分析的轨迹数据集」。

## 0. 理论问题

1. 轨迹的哪些字段是策略优化（Lab 45）必需的？
2. 无效动作、循环、截断各是什么行为缺陷的信号？
3. 失败分析如何区分「能力不足」与「协议误解」？

## 1. 数学基础

### 1.1 轨迹的数据结构

$$
\tau = (x,\ \{(a_t, o_t)\}_{t=1}^{K},\ a_{final},\ r,\ \text{meta})
$$

meta 含：逐步 token 序列与对数概率（Lab 45 的 ratio 计算需要 $\pi_{old}(a_t|h_t)$ 逐 token）、工具调用记录、耗时、截断标志。序列化设计原则：**轨迹文件自包含**（可离线重放：给定动作序列应重建同样观测——确定性环境；若环境随机则存种子）。

### 1.2 统计指标定义

- 任务成功率：$\mathbb E[r]$（Lab 03 的 MC 估计，标准误 $\sqrt{r(1-r)/n)}$——报告时带区间）;
- 动作合法率：合法动作步数 / 总步数；
- 无效动作率：按类型分桶（解析失败 / 未知工具 / 参数错误）；
- 循环检测：连续 $k$ 步动作字符串重复（$k \ge 2$ 判循环）；
- 工具效率：完成任务所需调用数 / 理论最少调用数。

### 1.3 失败模式分类树

```
失败
├── 截断型（T_max 用尽）
│   ├── 循环（同动作重复）        → 策略坍缩/上下文困惑
│   └── 无进展（动作合法但空转）   → 规划缺失
├── 终止型
│   ├── 答案错误但工具链正确       → 算术/检索能力不足
│   └── 答案错误且工具链错误       → 任务理解错误
└── 协议型（大量 INVALID）          → 格式遵循问题（SFT 不足）
```

分类树是 Lab 45 训练前后对比的观测框架：RL 改善哪一支、不动哪一支。

<iframe src="resources/figures/failure-taxonomy.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="失败分类树"></iframe>

<p align="center"><b>图 43.1</b>：失败 → 归因 → 修法</p>

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `record_trajectory` — 1.2 全部指标
- `batch_collect` — 1.2 全部指标
- `traj_stats` — 1.2 全部指标
- `dump_jsonl` — JSONL 落盘
- `classify_failure` — 1.3 分类树

- 接口骨架：[`../../code/lab43-trajectory/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab43-trajectory/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab43-trajectory/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab43-trajectory/selftest.py)
- 实验入口：`cd code/lab43-trajectory && uv sync && uv run run_lab43.py`


