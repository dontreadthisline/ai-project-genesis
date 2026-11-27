"""Lab 24 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 随机策略的迭代求值 vs 线性方程组直接解（allclose 1e-6）
# T2: 收敛速度：迭代误差比 ||v_{k+1}-v_k|| / ||v_k - v_{k-1}|| ≤ γ（slippery 下 ≈ γ）
# T3: policy_iteration 与 value_iteration 的 v* 一致（allclose 1e-6）
# T4: γ<1 时改进单调：每次改进后 v^{π'} ≥ v^π 逐状态成立
# T5: 停机界：ε-停机后 ||v-v*||∞ < ε（与更精确解对照）





# T1: 随机策略的迭代求值 vs 线性方程组直接解（allclose 1e-6）
# T2: 收敛速度：迭代误差比 ||v_{k+1}-v_k|| / ||v_k - v_{k-1}|| ≤ γ（slippery 下 ≈ γ）
# T3: policy_iteration 与 value_iteration 的 v* 一致（allclose 1e-6）
# T4: γ<1 时改进单调：每次改进后 v^{π'} ≥ v^π 逐状态成立
# T5: 停机界：ε-停机后 ||v-v*||∞ < ε（与更精确解对照）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
