"""Lab 18 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 可逆性：random x, f^{-1}(f(x)) == x（rel err < 1e-5）
# T2: log_det：单层耦合的解析值 exp(Σs) 与数值 Jacobian slogdet 一致（2x2 数值可行）
# T3: 密度归一化：网格上数值积分 log_prob 的 exp ≈ 1（容差 2%）
# T4: 目标分布解析对照：训练后模型在两峰附近密度高、在 (0,0) 附近密度低





# T1: 可逆性：random x, f^{-1}(f(x)) == x（rel err < 1e-5）
# T2: log_det：单层耦合的解析值 exp(Σs) 与数值 Jacobian slogdet 一致（2x2 数值可行）
# T3: 密度归一化：网格上数值积分 log_prob 的 exp ≈ 1（容差 2%）
# T4: 目标分布解析对照：训练后模型在两峰附近密度高、在 (0,0) 附近密度低



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
