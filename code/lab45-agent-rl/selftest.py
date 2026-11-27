"""Lab 45 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 轨迹 logprob 一致性：rollout 时记录的 logprob_old 与离线重算一致（1e-4）
# T2: 观测段梯度隔离：损失对观测 token 位置的 logprob 无梯度（mask 正确）
# T3: 更新前 ratio 全 1（重放同轨迹计算）
# T4: 组退化：同 prompt 全部轨迹同奖励时，该组参数更新量为 0
# T5: 评估独立性：evaluate 不接触任何奖励权重（真实判分专用）





# T1: 轨迹 logprob 一致性：rollout 时记录的 logprob_old 与离线重算一致（1e-4）
# T2: 观测段梯度隔离：损失对观测 token 位置的 logprob 无梯度（mask 正确）
# T3: 更新前 ratio 全 1（重放同轨迹计算）
# T4: 组退化：同 prompt 全部轨迹同奖励时，该组参数更新量为 0
# T5: 评估独立性：evaluate 不接触任何奖励权重（真实判分专用）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
