"""Lab 16 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 无偏性：两种估计对 mu 的梯度均值随 n=1e6 收敛到解析值（3 个标准误差内）
# T2: 方差比：n=1 时重复 5000 次的样本方差比 sf/pathwise（记录数量级）
# T3: baseline=b 的 sf：E[b * ∇log q] = 0（无偏不变，方差改变）
# T4: sigma 的梯度同构验证（解析 dE/dσ 存在时 pathwise 成立）





# T1: 无偏性：两种估计对 mu 的梯度均值随 n=1e6 收敛到解析值（3 个标准误差内）
# T2: 方差比：n=1 时重复 5000 次的样本方差比 sf/pathwise（记录数量级）
# T3: baseline=b 的 sf：E[b * ∇log q] = 0（无偏不变，方差改变）
# T4: sigma 的梯度同构验证（解析 dE/dσ 存在时 pathwise 成立）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
