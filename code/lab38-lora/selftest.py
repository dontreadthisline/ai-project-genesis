"""Lab 38 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 零初始化：B=0 时 LoRALinear 输出 == base 输出（逐位）
# T2: 参数量公式：2*d*r*层数*投影数 vs 实际可训练参数（误差 <1%）
# T3: 合并等价：merge 后的前向 == 未合并前向（1e-5）
# T4: 冻结检查：base 参数梯度恒为 0





# T1: 零初始化：B=0 时 LoRALinear 输出 == base 输出（逐位）
# T2: 参数量公式：2*d*r*层数*投影数 vs 实际可训练参数（误差 <1%）
# T3: 合并等价：merge 后的前向 == 未合并前向（1e-5）
# T4: 冻结检查：base 参数梯度恒为 0



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
