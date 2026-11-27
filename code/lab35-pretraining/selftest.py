"""Lab 35 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 初始 val loss ≈ ln(4000) ≈ 8.29（±0.1）
# T2: 梯度累积等价性：accum=2×8 与 batch=16 一步更新后参数差 < 1e-6（同数据）
# T3: PPL = exp(loss) 数值核对
# T4: checkpoint 恢复：加载后继续训练的 loss 与不中断连续训练一致（同种子）





# T1: 初始 val loss ≈ ln(4000) ≈ 8.29（±0.1）
# T2: 梯度累积等价性：accum=2×8 与 batch=16 一步更新后参数差 < 1e-6（同数据）
# T3: PPL = exp(loss) 数值核对
# T4: checkpoint 恢复：加载后继续训练的 loss 与不中断连续训练一致（同种子）



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
