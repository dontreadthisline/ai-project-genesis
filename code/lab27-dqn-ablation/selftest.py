"""Lab 27 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 偏差数值实验：动作数 k ∈ {2, 4, 8, 16}，噪声 N(0,1)，
#     E[max ε] 随 k 增大（MC 1e6 次），记录数值 vs 理论量级
# T2: Double 目标手算：给定两网络的 Q 值张量，验证选/评分离的计算
# T3: Dueling 恒等性：A 加常数 c、V 减 c 后 Q 不变
# T4: Dueling 输出与普通 QNet 接口一致（形状与 argmax 语义）





# T1: 偏差数值实验：动作数 k ∈ {2, 4, 8, 16}，噪声 N(0,1)，
#     E[max ε] 随 k 增大（MC 1e6 次），记录数值 vs 理论量级
# T2: Double 目标手算：给定两网络的 Q 值张量，验证选/评分离的计算
# T3: Dueling 恒等性：A 加常数 c、V 减 c 后 Q 不变
# T4: Dueling 输出与普通 QNet 接口一致（形状与 argmax 语义）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
