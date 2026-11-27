"""Lab 25 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 确定性检查：同种子同轨迹（FrozenLake 上 100 步逐位一致）
# T2: TD 更新手算：给定 Q 表与一步转移，验证单次更新数值
# T3: Q-Learning 在 Cliff Walking 学到最短路径（回报 ≈ -13），
#     SARSA 学到安全路径（回报 ≈ -17 ~ -25 区间、不贴崖）
# T4: ε=0（纯利用）+ 随机初始化：部分状态从未更新（探索失败的直接证据）





# T1: 确定性检查：同种子同轨迹（FrozenLake 上 100 步逐位一致）
# T2: TD 更新手算：给定 Q 表与一步转移，验证单次更新数值
# T3: Q-Learning 在 Cliff Walking 学到最短路径（回报 ≈ -13），
#     SARSA 学到安全路径（回报 ≈ -17 ~ -25 区间、不贴崖）
# T4: ε=0（纯利用）+ 随机初始化：部分状态从未更新（探索失败的直接证据）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
