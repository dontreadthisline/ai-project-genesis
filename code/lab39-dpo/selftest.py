"""Lab 39 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 平局退化：y_w == y_l 时 loss == ln 2 ≈ 0.693（ratio 差为 0 → σ(0)）
# T2: 极限行为：π == ref 时 loss == ln 2；π(w)↑ π(l)↓ 时 loss → 0
# T3: 隐式奖励公式：用 1.2 反解式数值验证训练中的 r̂ 单调性
# T4: β 扫描的梯度范数：β 大则更新小（同一 batch）





# T1: 平局退化：y_w == y_l 时 loss == ln 2 ≈ 0.693（ratio 差为 0 → σ(0)）
# T2: 极限行为：π == ref 时 loss == ln 2；π(w)↑ π(l)↓ 时 loss → 0
# T3: 隐式奖励公式：用 1.2 反解式数值验证训练中的 r̂ 单调性
# T4: β 扫描的梯度范数：β 大则更新小（同一 batch）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
