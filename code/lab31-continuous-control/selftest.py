"""Lab 31 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 确定性策略梯度与数值梯度一致（rel err < 1e-4，小网络）
# T2: 双 Q min 的减偏：对照单 Q 目标，训练后 Q 估计 vs 实际回报偏差
# T3: tanh 压缩 log_prob：包含 -Σ log(1 - tanh(u)^2) 修正项（无此项则概率不归一）
# T4: 延迟更新计数：actor 参数在非更新步严格不变





# T1: 确定性策略梯度与数值梯度一致（rel err < 1e-4，小网络）
# T2: 双 Q min 的减偏：对照单 Q 目标，训练后 Q 估计 vs 实际回报偏差
# T3: tanh 压缩 log_prob：包含 -Σ log(1 - tanh(u)^2) 修正项（无此项则概率不归一）
# T4: 延迟更新计数：actor 参数在非更新步严格不变



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
