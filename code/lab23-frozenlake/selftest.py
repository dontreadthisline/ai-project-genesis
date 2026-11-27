"""Lab 23 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 每行概率和为 1（所有 (s,a)）
# T2: 终止态自吸收：P[hole, a, hole] == 1
# T3: slippery 概率分布：非边界状态上三个可能转移各 1/3
#     边界/角落撞墙时分布折叠（概率合并回合法状态）
# T4: sample_step 与 P 的频率一致（1e5 次采样，各 s' 频率误差 < 1%）





# T1: 每行概率和为 1（所有 (s,a)）
# T2: 终止态自吸收：P[hole, a, hole] == 1
# T3: slippery 概率分布：非边界状态上三个可能转移各 1/3
#     边界/角落撞墙时分布折叠（概率合并回合法状态）
# T4: sample_step 与 P 的频率一致（1e5 次采样，各 s' 频率误差 < 1%）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
