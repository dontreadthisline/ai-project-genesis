"""Lab 19 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 边缘统计：固定 x0，采样 q(x_t|x0) 1e4 次，均值 ≈ sqrt(abar_t)*x0、
#     方差 ≈ 1-abar_t（容差 3%）
# T2: t=T 时 abar_T ≈ 0（x_T 近似标准正态，与 x0 无关）
# T3: 训练收敛后 loss < 0.1 量级（MNIST 小模型）
# T4: 噪声预测 vs x0 预测等价性：eps_hat 换算 x0_hat = (x_t - sqrt(1-abar)*eps)/sqrt(abar)





# T1: 边缘统计：固定 x0，采样 q(x_t|x0) 1e4 次，均值 ≈ sqrt(abar_t)*x0、
#     方差 ≈ 1-abar_t（容差 3%）
# T2: t=T 时 abar_T ≈ 0（x_T 近似标准正态，与 x0 无关）
# T3: 训练收敛后 loss < 0.1 量级（MNIST 小模型）
# T4: 噪声预测 vs x0 预测等价性：eps_hat 换算 x0_hat = (x_t - sqrt(1-abar)*eps)/sqrt(abar)



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
