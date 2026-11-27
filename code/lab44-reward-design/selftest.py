"""Lab 44 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 纯 outcome：正确轨迹 R=1、错误 R=0、截断 R=0
# T2: 成本惩罚单调性：同任务两条轨迹（3 步 vs 10 步均成功），R 差 == w_c*7
# T3: hack 复现 1：process 按调用计分时，「空转刷工具」轨迹的 R 超过直接答对的最短轨迹
# T4: hack 复现 2：格式奖励含关键词时，复读关键词的轨迹得高分
# T5: 权重为零的退化：w_o=1, 其余 0 时 R == outcome（还原 Lab 43 基线）





# T1: 纯 outcome：正确轨迹 R=1、错误 R=0、截断 R=0
# T2: 成本惩罚单调性：同任务两条轨迹（3 步 vs 10 步均成功），R 差 == w_c*7
# T3: hack 复现 1：process 按调用计分时，「空转刷工具」轨迹的 R 超过直接答对的最短轨迹
# T4: hack 复现 2：格式奖励含关键词时，复读关键词的轨迹得高分
# T5: 权重为零的退化：w_o=1, 其余 0 时 R == outcome（还原 Lab 43 基线）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
