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

### 1.2 上下文拼接协议

每步上下文 = 系统提示 + 任务 + 历史轮次（THOUGHT / ACTION / OBSERVATION 标记化拼接）。协议即「观测编码」：格式必须无歧义（模型能区分自己说的话与工具返回），且训练与推理的拼接格式严格一致（Lab 45 的 rollouts 用同一协议）。

### 1.3 动作空间

三种工具动作 + 终止：

- `calc(expr)`：安全表达式求值（白名单算符），非法输入返回错误字符串（不抛异常——错误也是 observation）；
- `query(key)`：预置结构化数据（Python 字典/SQLite 表）按键查值，查无返回 NOT_FOUND；
- `py(expr)`：受限执行器：`eval` 于只含白名单内建函数的命名空间，禁 `import/open/exec/__`，设超时；
- `answer(x)` / `finish`：终止并给出最终答案。

无效动作（解析失败、未知工具名）：环境返回 `INVALID_ACTION` 提示并继续——这是 Lab 43 无效动作统计的来源。**环境的鲁棒性原则：任何模型输出都必须有定义好的响应**。

### 1.4 部分可观测性

Agent 只见观测不见环境内部状态（如数据表全貌）：这是 POMDP。策略依赖历史 $h_t$ 而非当前状态 $s_t$——上下文窗口就是「信念状态」的载体。

## 2. 实现任务

```python
class ToolEnv:
    def __init__(self, tasks): ...
    def reset(self) -> str: ...               # 返回任务 prompt
    def step(self, action_str) -> (obs, done, info): ...
    def score(self, final_answer) -> float: ...  # 可验证判分
def agent_loop(model, env, max_steps, sampler): ...
    # 拼接 -> 解码动作段 -> env.step -> 拼 observation -> 循环
```

任务族（可自动生成 + 判分）：算术（calc 可全对）、查表问答（query）、两者组合的多步任务（先查数再算）。

自测：

```python
# T1: 工具边界：calc("__import__('os')") 返回错误串而非执行；py 超时返回 TIMEOUT
# T2: 环境全响应：对 20 种任意乱码 action，env.step 均返回定义好的观测、不抛异常
# T3: 终止语义：answer 后 done=True；max_steps 截断 done=True 且 info 标记 truncated
# T4: 上下文一致性：同一交互序列两次拼接的 token 序列完全一致（确定性协议）
# T5: 判分器：构造对/错/格式错答案，score ∈ {1, 0} 边界清晰
```

## 3. 对照实验

| 编号 | 自变量 | 控制变量 | 因变量 |
|---|---|---|---|
| E1 | prompt 中工具文档：完整/缺失/错误 | SFT 前 | 动作合法率、成功率 |
| E2 | $T_{max} \in \{3, 8, 20\}$ | — | 截断率、成功率的边际 |
| E3 | 观测格式：结构化标记 vs 裸文本 | — | 模型利用观测的能力 |
| E4 | 任务难度：单步 / 两步 / 三步 | — | 成功率衰减（组合误差） |

## 4. 预期图表

- 图 1：一个完整轨迹的逐 token 展示（THOUGHT/ACTION/OBSERVATION 着色）。
- 图 2：E1 三种文档水平的动作合法率柱状。
- 图 3：E4 成功率 vs 步骤数（指数衰减拟合，每步成功率 $p$ 的组合）。
- 图 4：E2 的截断-成功率权衡曲线。

## 5. 观测指标

`task_success_rate`、`action_validity`、`avg_steps`、`truncation_rate`、`tool_call_freq`。

## 6. 结果分析

1. E4 的衰减率隐含的单步成功率是多少？$0.9^3 \approx 0.73$——多步任务是 Lab 45 要用 RL 修复的核心瓶颈。
2. E3 中结构化观测为什么更有效？（边界清晰 = 上下文分布更可预测。）
3. 本实验的策略是冻结的 SFT 模型——它失败在哪里（不会规划/不会纠错），RL 阶段能修复哪一部分？

## 7. 验收清单

- [ ] T1-T5 通过；三个工具 + 终止动作 + 全响应鲁棒性
- [ ] 能画出一个完整交互的状态-动作-观测序列图
- [ ] 明确本阶段产物（env + 协议 + 判分器）将原封不动进入 Lab 45
