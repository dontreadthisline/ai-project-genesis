"""Lab 29 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: δ 的期望 = 优势：表格 MDP（Lab 23 的 P/R）上，收敛 V 后验证
#     E_{a}[δ(s,a)] ≈ 0（均值）且 E[δ | s, a*] > E[δ | s, a_worst]
# T2: detach 检查：actor 损失对 V 参数梯度恒为 0
# T3: A 的符号逻辑：构造两条动作轨迹（好坏），δ 符号与动作质量一致
# T4: 训练后 V(s) 与实际折扣回报的 MC 估计一致（容差 15%）





# T1: δ 的期望 = 优势：表格 MDP（Lab 23 的 P/R）上，收敛 V 后验证
#     E_{a}[δ(s,a)] ≈ 0（均值）且 E[δ | s, a*] > E[δ | s, a_worst]
# T2: detach 检查：actor 损失对 V 参数梯度恒为 0
# T3: A 的符号逻辑：构造两条动作轨迹（好坏），δ 符号与动作质量一致
# T4: 训练后 V(s) 与实际折扣回报的 MC 估计一致（容差 15%）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
