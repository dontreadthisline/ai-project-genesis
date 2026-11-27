"""Lab 09 自测断言（红灯标准: 实现完成前全部以 NotImplementedError 失败）。"""


# T1: 卷积前向与朴素双循环参考实现 allclose（随机 3x3 输入）
# T2: Conv2d 反向与数值梯度一致（rel err < 1e-6）
# T3: 输出尺寸公式：28x28, K=5, s=1, p=2 -> 28x28；s=2, p=0 -> 12x12
# T4: 参数量手算与实现一致：conv5x5(6,1) = 156；fc(120, 16*4*4) = 30840





# T1: 卷积前向与朴素双循环参考实现 allclose（随机 3x3 输入）
# T2: Conv2d 反向与数值梯度一致（rel err < 1e-6）
# T3: 输出尺寸公式：28x28, K=5, s=1, p=2 -> 28x28；s=2, p=0 -> 12x12
# T4: 参数量手算与实现一致：conv5x5(6,1) = 156；fc(120, 16*4*4) = 30840



def check_implementation() -> None:
    """把上方讲义自测条目转成断言并在此调用；全部通过后返回。"""
    raise NotImplementedError("TODO: 实现讲义自测断言")
