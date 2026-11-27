"""Lab 28 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: baseline 无偏性数值验证：固定状态、随机 b，E[b ∇logπ] 随采样数 → 0
#     （1e5 样本，均值范数 < 1e-2 × 无 b 时的梯度范数）
# T2: 梯度方向检查：增大高回报动作的 logπ 概率（单步手工例）
# T3: 方差对照：同一批轨迹，带/不带 baseline 的梯度估计样本方差比值
# T4: 收敛后策略接近确定性：eval 时动作熵 < 0.3 nats





# T1: baseline 无偏性数值验证：固定状态、随机 b，E[b ∇logπ] 随采样数 → 0
#     （1e5 样本，均值范数 < 1e-2 × 无 b 时的梯度范数）
# T2: 梯度方向检查：增大高回报动作的 logπ 概率（单步手工例）
# T3: 方差对照：同一批轨迹，带/不带 baseline 的梯度估计样本方差比值
# T4: 收敛后策略接近确定性：eval 时动作熵 < 0.3 nats



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
