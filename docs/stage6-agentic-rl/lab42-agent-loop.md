---
title: "Lab 42：实现基础 Agent"
lab: 42
stage: 6
---

对应大纲实验 42。本阶段起点：把第五阶段的「语言模型 + 解码」扩展为「语言模型 + 工具交互闭环」。

## 0. 理论问题

1. Agent loop 与标准解码循环的差别在哪一步？
2. Observation 如何进入上下文？拼接格式决定什么？
3. 受限工具的安全性边界如何设计？

## 1. 数学基础

### 1.1 Agent loop

$$
\cdots \to \underbrace{\text{Observation}}_{\text{环境返回}} \to \underbrace{\text{Policy } \pi_\theta(a \mid h_t)}_{\text{LM 解码}} \to \underbrace{\text{Action}}_{\text{工具调用/回答}} \to \underbrace{\text{Tool 执行}}_{\text{环境转移}} \to \cdots
$$

$h_t$ 是对话历史（prompt + 历次动作与观察的拼接）。与 Lab 36 的纯解码差别：序列不再一次性由策略生成——环境在中间注入 token（observation）。策略分布定义在「动作段」上，观察段是给定条件。终止条件：模型发出终止动作、或达到最大步数 $T_{max}$（防死循环）。

<iframe src="resources/figures/agent-cycle.html" style="width:100%;height:384px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="Agent Loop：闭环"></iframe>

<p align="center"><b>图 42.1</b>：obs → policy → action → tool 循环</p>

### 1.2 上下文拼接协议

每步上下文 = 系统提示 + 任务 + 历史轮次（THOUGHT / ACTION / OBSERVATION 标记化拼接）。协议即「观测编码」：格式必须无歧义（模型能区分自己说的话与工具返回），且训练与推理的拼接格式严格一致（Lab 45 的 rollouts 用同一协议）。

### 1.3 动作空间

三种工具动作 + 终止：

- `calc(expr)`：安全表达式求值（白名单算符），非法输入返回错误字符串（不抛异常——错误也是 observation）；
- `query(key)`：预置结构化数据（Python 字典/SQLite 表）按键查值，查无返回 NOT_FOUND；
- `py(expr)`：受限执行器：`eval` 于只含白名单内建函数的命名空间，禁 `import/open/exec/__`，设超时；
- `answer(x)` / `finish`：终止并给出最终答案。

无效动作（解析失败、未知工具名）：环境返回 `INVALID_ACTION` 提示并继续——这是 Lab 43 无效动作统计的来源。**环境的鲁棒性原则：任何模型输出都必须有定义好的响应**。

<iframe src="resources/figures/tool-safety.html" style="width:100%;height:354px;border:1px solid rgba(58,74,168,.18);border-radius:8px" title="受限工具的边界"></iframe>

<p align="center"><b>图 42.2</b>：安全边界（进得来的才执行）</p>

### 1.4 部分可观测性

Agent 只见观测不见环境内部状态（如数据表全貌）：这是 POMDP。策略依赖历史 $h_t$ 而非当前状态 $s_t$——上下文窗口就是「信念状态」的载体。

## 2. 实现任务

实现清单（签名与 TODO 见骨架源码）：

- `ToolEnv` — 返回任务 prompt
- `agent_loop` — 拼接 -> 解码动作段 -> env.step -> 拼 observation -> 循环

- 接口骨架：[`../../code/lab42-agent-loop/skeleton.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab42-agent-loop/skeleton.py)
- 自测断言（红灯标准，实现完成前应全部失败）：[`../../code/lab42-agent-loop/selftest.py`](https://github.com/dontreadthisline/ai-project-genesis/blob/main/code/lab42-agent-loop/selftest.py)
- 实验入口：`cd code/lab42-agent-loop && uv sync && uv run run_lab42.py`


